"""
train_nutrient.py
Training pipeline for Model 2: Potassium Deficiency Classification.
Classes: Potassium_Deficiency vs Normal_Status
Architecture: MobileNetV3-Large (Transfer Learning from ImageNet)
Dataset: raw/potassium_deficiency + Audited Healthy Controls
Strictly avoids claiming Nitrogen or Phosphorus deficiency detection.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
DATA_DIR = ROOT_DIR / "data"
META_PATH = DATA_DIR / "metadata" / "nutrient_metadata.csv"
RUNS_DIR = ROOT_DIR / "runs" / "nutrient_v1"
MODEL_DIR = ROOT_DIR / "models" / "nutrient"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal_Status", "Potassium_Deficiency"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class NutrientDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CLASS_TO_IDX[row["class_name"]]
        return img, label, row["dataset_source"], row["image_id"]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes=2):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test for 1 epoch")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    epochs = 1 if args.smoke else args.epochs

    print("=" * 60)
    print("MODEL 2 — POTASSIUM DEFICIENCY CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    df = pd.read_csv(META_PATH)
    print(f"Dataset Manifest Loaded: {len(df)} records from {META_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("NOTE: Nitrogen and Phosphorus deficiency are NOT supported by dataset.")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Split counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print("\nClass distribution in Train:")
    for c, cnt in train_df["class_name"].value_counts().items():
        print(f"  {c:<25}: {cnt}")

    # Class weights for binary imbalance (1034 K vs 301 Normal)
    counts = [train_df["class_name"].value_counts().get(c, 1) for c in CLASS_NAMES]
    total = len(train_df)
    weights = [total / (2.0 * cnt) for cnt in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class Weights for Loss: Normal={weights[0]:.2f}, K_Def={weights[1]:.2f}")

    train_transform, val_transform = get_transforms()
    train_dataset = NutrientDataset(train_df, transform=train_transform)
    val_dataset = NutrientDataset(val_df, transform=val_transform)
    test_dataset = NutrientDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_epoch = 0
    best_checkpoint_path = MODEL_DIR / "best.pt"
    history = []

    print("\n" + "="*60)
    print(f"Starting Training: {epochs} epoch(s), batch_size={args.batch_size}, lr={args.lr}")
    print("="*60)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        val_eval = evaluate(model, val_loader, criterion, device)
        val_loss = val_eval["loss"]
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_roc_auc": round(val_eval["roc_auc"], 4)
        })

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_eval['roc_auc']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "classes": CLASS_NAMES,
                "architecture": "MobileNetV3-Large"
            }, best_checkpoint_path)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f}s. Best epoch: {best_epoch} with Val F1: {best_val_f1:.4f}")

    # Pristine test evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON PRISTINE TEST SET")
    print("="*60)
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_eval = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy:  {test_eval['accuracy']*100:.2f}%")
    print(f"Test Precision: {test_eval['precision']:.4f}")
    print(f"Test Recall:    {test_eval['recall']:.4f}")
    print(f"Test F1-Score:  {test_eval['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_eval['roc_auc']:.4f}")
    print(f"Confusion Matrix [Normal, Potassium_Deficiency]:")
    print(np.array(test_eval["confusion_matrix"]))

    # Save run experiment tracking artifacts
    pd.DataFrame(history).to_csv(RUNS_DIR / "training_log.csv", index=False)

    experiment_summary = {
        "task": "potassium_deficiency_classification",
        "model_architecture": "MobileNetV3-Large",
        "num_classes": 2,
        "classes": CLASS_NAMES,
        "dataset_version": "v1_potassium_and_healthy_controls",
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "AdamW",
            "loss": "ClassWeightedCrossEntropy",
            "seed": SEED,
            "input_resolution": [224, 224]
        },
        "training_time_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "validation_best_f1": round(best_val_f1, 4),
        "test_metrics": {
            "accuracy": round(float(test_eval["accuracy"]), 4),
            "precision": round(float(test_eval["precision"]), 4),
            "recall": round(float(test_eval["recall"]), 4),
            "f1": round(float(test_eval["f1"]), 4),
            "roc_auc": round(float(test_eval["roc_auc"]), 4)
        },
        "confusion_matrix": test_eval["confusion_matrix"]
    }

    with open(RUNS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)

    # Export to ONNX
    print("\n" + "="*60)
    print("EXPORTING TO ONNX")
    print("="*60)
    onnx_path = MODEL_DIR / "model.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14
        )
        print(f"Exported ONNX model successfully to: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"ONNX export warning: {e}")

    print("\nMODEL 2 NUTRIENT TRAINING & EXPORT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
