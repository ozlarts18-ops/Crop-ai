"""
train_disease.py
Training pipeline for locked Model 1: Disease Classification
Architecture: EfficientNetV2-M
Classes (9):
  0: Bacterial_Blight
  1: Cercospora_Leaf_Blight
  2: Frogeye_Leaf_Spot
  3: Healthy
  4: Mosaic
  5: Pest_Damage
  6: Septoria_Brown_Spot
  7: Soybean_Rust
  8: Sudden_Death_Syndrome

Features:
- Stage 1: Classifier head warmup (frozen backbone)
- Stage 2: Backbone unfreezing & full fine-tuning
- AdamW optimizer, CosineAnnealingLR, Label Smoothing (0.08)
- AMP (Automatic Mixed Precision)
- Early stopping on Validation Macro F1
- Evaluation on untouched test set: Accuracy, Macro F1, Weighted F1, Precision, Recall, Per-class Recall, Confusion Matrix
- Checkpoint saving & ONNX export
"""

import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
CONFIG = {
    "model_name": "EfficientNetV2-M",
    "task": "disease_classification",
    "num_classes": 9,
    "image_size": 224,
    "batch_size": 32,
    "stage1_epochs": 5,
    "stage1_lr": 1e-3,
    "stage2_epochs": 25,
    "stage2_lr": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "label_smoothing": 0.08,
    "patience": 7,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# Load label mapping
with open("models/labels/disease_labels.json", "r") as f:
    label_map_raw = json.load(f)
label_to_idx = {v: int(k) for k, v in label_map_raw.items()}
idx_to_label = {int(k): v for k, v in label_map_raw.items()}
class_names = [idx_to_label[i] for i in range(CONFIG["num_classes"])]

# -------------------------------------------------------------
# Dataset & Transforms
# -------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LeafDiseaseDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            c_name = row["class_name"]
            if c_name in label_to_idx:
                self.samples.append((row["filepath"], label_to_idx[c_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------------------------------------
# Evaluation Helper
# -------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Per-class recall
    _, per_class_recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, labels=list(range(CONFIG["num_classes"])), zero_division=0)
    
    return {
        "loss": avg_loss,
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))},
        "all_preds": all_preds,
        "all_targets": all_targets
    }

# -------------------------------------------------------------
# Main Training Function
# -------------------------------------------------------------
def main():
    print(f"=== Starting Training: {CONFIG['model_name']} ({CONFIG['task']}) ===")
    device = CONFIG["device"]
    print(f"Device: {device}")

    # Load data
    train_ds = LeafDiseaseDataset("data/splits/disease_train.csv", transform=train_transforms)
    val_ds = LeafDiseaseDataset("data/splits/disease_val.csv", transform=eval_transforms)
    test_ds = LeafDiseaseDataset("data/splits/disease_test.csv", transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # Initialize model
    print("Loading pretrained EfficientNetV2-M weights...")
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    model = models.efficientnet_v2_m(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, CONFIG["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # Tracking records
    os.makedirs("experiments/disease", exist_ok=True)
    log_records = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    # ---------------------------------------------------------
    # STAGE 1: Train classification head (Backbone frozen)
    # ---------------------------------------------------------
    print("\n--- STAGE 1: Classification Head Warmup (Backbone Frozen) ---")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["stage1_lr"], weight_decay=CONFIG["weight_decay"])

    for epoch in range(1, CONFIG["stage1_epochs"] + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"Stage 1 - Epoch [{epoch}/{CONFIG['stage1_epochs']}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}", flush=True)

        log_records.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "lr": CONFIG["stage1_lr"]
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), "experiments/disease/best_stage1.pt")
            torch.save(model.state_dict(), "models/checkpoints/disease_efficientnetv2_m_best.pt")

    # ---------------------------------------------------------
    # STAGE 2: Full backbone fine-tuning
    # ---------------------------------------------------------
    print("\n--- STAGE 2: Full Backbone Fine-Tuning ---", flush=True)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["stage2_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["stage2_epochs"], eta_min=CONFIG["min_lr"])

    for epoch in range(1, CONFIG["stage2_epochs"] + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        current_lr = scheduler.get_last_lr()[0]

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        overall_epoch = CONFIG["stage1_epochs"] + epoch
        print(f"Stage 2 - Epoch [{epoch}/{CONFIG['stage2_epochs']}] ({elapsed:.1f}s) - LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}", flush=True)

        log_records.append({
            "stage": 2,
            "epoch": overall_epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "lr": current_lr
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = overall_epoch
            patience_counter = 0
            print(f"  --> Best checkpoint improved! Val Macro F1: {best_val_macro_f1:.4f} (Epoch {best_epoch}). Saving...", flush=True)
            torch.save(model.state_dict(), "experiments/disease/best_checkpoint.pt")
            torch.save(model.state_dict(), "models/checkpoints/disease_efficientnetv2_m_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement on Val Macro F1.", flush=True)
                break

    # Save training log
    pd.DataFrame(log_records).to_csv("experiments/disease/training_log.csv", index=False)
    print("\nTraining log saved to experiments/disease/training_log.csv")

    # ---------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET
    # ---------------------------------------------------------
    print("\n--- Running Final Evaluation on Untouched Test Set ---")
    best_weights_path = "models/checkpoints/disease_efficientnetv2_m_best.pt"
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nFINAL TEST RESULTS (Best Epoch: {best_epoch}):")
    print(f"  Test Accuracy:     {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"  Test Precision:    {test_metrics['macro_precision']:.4f}")
    print(f"  Test Recall:       {test_metrics['macro_recall']:.4f}")
    print("\nPer-Class Recall:")
    for k, v in test_metrics["per_class_recall"].items():
        print(f"  - {k}: {v * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(test_metrics["all_targets"], test_metrics["all_preds"], labels=list(range(CONFIG["num_classes"])))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Disease Classification Confusion Matrix (Test Acc: {test_metrics['accuracy']*100:.1f}%)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig("experiments/disease/confusion_matrix.png", dpi=300)
    plt.close()

    # Save metrics.json & config.yaml
    final_results = {
        "model": CONFIG["model_name"],
        "task": CONFIG["task"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "per_class_recall": test_metrics["per_class_recall"],
        "confusion_matrix": cm.tolist()
    }
    with open("experiments/disease/metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    with open("experiments/disease/config.yaml", "w") as f:
        yaml.dump(CONFIG, f)

    # ---------------------------------------------------------
    # ONNX EXPORT
    # ---------------------------------------------------------
    print("\n--- Exporting Model to ONNX ---")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"])
    onnx_path = "models/exports/disease_efficientnetv2_m.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX exported successfully to {onnx_path}")

    # Validate ONNX with onnxruntime
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"ONNXRuntime inference check passed! Output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    main()
