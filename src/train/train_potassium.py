"""
train_potassium.py
Training pipeline for locked Model 2: Potassium Deficiency Detection
Architecture: EfficientNetV2-S
Classes (2):
  0: Normal_Status
  1: Potassium_Deficiency

Features:
- Stage 1: Classifier head warmup (frozen backbone)
- Stage 2: Backbone unfreezing & full fine-tuning
- AdamW optimizer, CosineAnnealingLR
- AMP (Automatic Mixed Precision)
- Validation threshold tuning over [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  prioritizing Potassium_Deficiency recall
- Threshold locked to models/labels/potassium_threshold.json
- Final untouched test evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
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
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt

CONFIG = {
    "model_name": "EfficientNetV2-S",
    "task": "potassium_deficiency_detection",
    "num_classes": 2,
    "image_size": 224,
    "batch_size": 32,
    "stage1_epochs": 5,
    "stage1_lr": 1e-3,
    "stage2_epochs": 20,
    "stage2_lr": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "patience": 6,
    "seed": 42,
    "candidate_thresholds": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

with open("models/labels/potassium_labels.json", "r") as f:
    label_map_raw = json.load(f)
label_to_idx = {v: int(k) for k, v in label_map_raw.items()}
class_names = [label_map_raw[str(i)] for i in range(2)]

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

class PotassiumDataset(Dataset):
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

def get_probabilities(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    return np.array(all_probs), np.array(all_targets)

def evaluate_with_threshold(probs, targets, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=1, zero_division=0)
    
    # Per-class recall
    _, per_class_recall, _, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1], zero_division=0)
    
    try:
        roc_auc = roc_auc_score(targets, probs)
    except Exception:
        roc_auc = 0.5

    prec_arr, rec_arr, _ = precision_recall_curve(targets, probs, pos_label=1)
    pr_auc = auc(rec_arr, prec_arr)

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "normal_recall": float(per_class_recall[0]),
        "potassium_recall": float(per_class_recall[1]),
        "threshold": float(threshold),
        "preds": preds.tolist(),
        "targets": targets.tolist()
    }

def main():
    print(f"=== Starting Training: {CONFIG['model_name']} ({CONFIG['task']}) ===")
    device = CONFIG["device"]
    print(f"Device: {device}")

    train_ds = PotassiumDataset("data/splits/potassium_train.csv", transform=train_transforms)
    val_ds = PotassiumDataset("data/splits/potassium_val.csv", transform=eval_transforms)
    test_ds = PotassiumDataset("data/splits/potassium_test.csv", transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    print("Loading pretrained EfficientNetV2-S weights...")
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, CONFIG["num_classes"])
    model = model.to(device)

    # Weighted loss if needed, or CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    os.makedirs("experiments/potassium", exist_ok=True)
    log_records = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    # ---------------------------------------------------------
    # STAGE 1: Classifier head warmup
    # ---------------------------------------------------------
    print("\n--- STAGE 1: Head Warmup (Backbone Frozen) ---")
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
        val_probs, val_targets = get_probabilities(model, val_loader, device)
        val_metrics = evaluate_with_threshold(val_probs, val_targets, threshold=0.5)
        elapsed = time.time() - start_time

        print(f"Stage 1 - Epoch [{epoch}/{CONFIG['stage1_epochs']}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | K-Recall: {val_metrics['potassium_recall']:.4f}", flush=True)

        log_records.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_k_recall": val_metrics["potassium_recall"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": CONFIG["stage1_lr"]
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), "experiments/potassium/best_stage1.pt")
            torch.save(model.state_dict(), "models/checkpoints/potassium_efficientnetv2_s_best.pt")

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
        val_probs, val_targets = get_probabilities(model, val_loader, device)
        val_metrics = evaluate_with_threshold(val_probs, val_targets, threshold=0.5)
        elapsed = time.time() - start_time

        overall_epoch = CONFIG["stage1_epochs"] + epoch
        print(f"Stage 2 - Epoch [{epoch}/{CONFIG['stage2_epochs']}] ({elapsed:.1f}s) - LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | K-Recall: {val_metrics['potassium_recall']:.4f}", flush=True)

        log_records.append({
            "stage": 2,
            "epoch": overall_epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_k_recall": val_metrics["potassium_recall"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": current_lr
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = overall_epoch
            patience_counter = 0
            print(f"  --> Best checkpoint saved! Val F1: {best_val_f1:.4f} (Epoch {best_epoch})", flush=True)
            torch.save(model.state_dict(), "experiments/potassium/best_checkpoint.pt")
            torch.save(model.state_dict(), "models/checkpoints/potassium_efficientnetv2_s_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.", flush=True)
                break

    pd.DataFrame(log_records).to_csv("experiments/potassium/training_log.csv", index=False)

    # ---------------------------------------------------------
    # THRESHOLD TUNING ON VALIDATION SET ONLY
    # ---------------------------------------------------------
    print("\n--- Tuning Classification Threshold on Validation Data ---")
    best_weights_path = "models/checkpoints/potassium_efficientnetv2_s_best.pt"
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    val_probs, val_targets = get_probabilities(model, val_loader, device)

    threshold_results = []
    selected_threshold = 0.50
    best_thresh_score = -1.0

    print("Threshold Evaluation on Val Set:")
    for th in CONFIG["candidate_thresholds"]:
        res = evaluate_with_threshold(val_probs, val_targets, threshold=th)
        threshold_results.append({
            "threshold": th,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "k_recall": res["potassium_recall"],
            "normal_recall": res["normal_recall"]
        })
        print(f"  Th: {th:.2f} | Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | K-Recall: {res['potassium_recall']:.4f} | F1: {res['f1']:.4f}")
        # Prioritize Potassium_Deficiency recall with strong F1
        score = res["f1"] * 0.5 + res["potassium_recall"] * 0.5
        if score > best_thresh_score:
            best_thresh_score = score
            selected_threshold = th

    print(f"\n--> Selected Optimal Threshold: {selected_threshold:.2f}")
    threshold_data = {
        "model": "EfficientNetV2-S",
        "task": "Potassium_Deficiency",
        "selected_threshold": selected_threshold,
        "validation_evaluations": threshold_results
    }
    with open("models/labels/potassium_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    print("Threshold saved to models/labels/potassium_threshold.json")

    # ---------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET
    # ---------------------------------------------------------
    print(f"\n--- Running Final Evaluation on Untouched Test Set (Threshold = {selected_threshold:.2f}) ---")
    test_probs, test_targets = get_probabilities(model, test_loader, device)
    test_metrics = evaluate_with_threshold(test_probs, test_targets, threshold=selected_threshold)

    print(f"\nFINAL TEST RESULTS (Best Epoch: {best_epoch}):")
    print(f"  Test Accuracy:     {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Precision:    {test_metrics['precision']:.4f}")
    print(f"  Test Recall:       {test_metrics['recall']:.4f} (Potassium Recall)")
    print(f"  Test Normal Recall:{test_metrics['normal_recall']:.4f}")
    print(f"  Test F1-Score:     {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:       {test_metrics['pr_auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_targets, test_metrics["preds"], labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Potassium Deficiency Confusion Matrix (Th={selected_threshold:.2f}, Acc={test_metrics['accuracy']*100:.1f}%)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=11)

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig("experiments/potassium/confusion_matrix.png", dpi=300)
    plt.close()

    # Save metrics.json & config.yaml
    final_results = {
        "model": CONFIG["model_name"],
        "task": CONFIG["task"],
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_normal_recall": test_metrics["normal_recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "confusion_matrix": cm.tolist()
    }
    with open("experiments/potassium/metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    with open("experiments/potassium/config.yaml", "w") as f:
        yaml.dump(CONFIG, f)

    # ---------------------------------------------------------
    # ONNX EXPORT
    # ---------------------------------------------------------
    print("\n--- Exporting Model to ONNX ---")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"])
    onnx_path = "models/exports/potassium_efficientnetv2_s.onnx"
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

    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"ONNXRuntime inference check passed! Output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    main()
