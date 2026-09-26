"""
train_crop_verifier.py
Crop Verification Model Training Pipeline
Model: EfficientNetV2-S
Task: Binary Crop Verification (Soybean vs Non_Soybean)
Dataset: SoyCotton (Soybean: 577, Cotton/Non_Soybean: 577)

Classes:
  0: Soybean
  1: Non_Soybean

Pipeline:
  1. Stage 1: Classification Head Warmup (5 epochs, frozen backbone)
  2. Stage 2: Full Fine-Tuning (up to 20 epochs, unfrozen backbone, CosineAnnealingLR, AdamW, AMP)
  3. Validation-based Checkpoint Selection (Primary metric: Validation F1)
  4. Validation-only Threshold Tuning ([0.30 - 0.70])
  5. Single Final Test on Untouched Test Set
  6. ONNX Export & ONNXRuntime Verification
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
    "task": "crop_verification",
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

# Load label mapping
LABEL_MAP_PATH = "models/labels/crop_verification_labels.json"
os.makedirs("models/labels", exist_ok=True)
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
idx_to_label = {int(k): v for k, v in label_map.items()}
label_to_idx = {v: int(k) for k, v in label_map.items()}
class_names = [idx_to_label[0], idx_to_label[1]]

# Realistic crop augmentations (deterministic for val/test)
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

class CropVerificationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            c_name = row["class_name_binary"]
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
    """Returns probability of Soybean (class 0) and ground truth targets."""
    model.eval()
    all_probs_soy = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 0] # probability of Soybean
            all_probs_soy.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    return np.array(all_probs_soy), np.array(all_targets)

def compute_metrics(probs_soy, targets, threshold=0.5):
    """
    Evaluates predictions where class 0 is Soybean and class 1 is Non_Soybean.
    A sample is predicted as Soybean if probs_soy >= threshold.
    """
    preds = np.where(probs_soy >= threshold, 0, 1)
    acc = accuracy_score(targets, preds)
    
    # Soybean is label 0
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=0, zero_division=0)
    
    # Per-class recall: [Soybean_recall, Non_Soybean_recall]
    _, per_class_recall, _, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1], zero_division=0)
    soybean_recall = float(per_class_recall[0])
    non_soybean_recall = float(per_class_recall[1])
    
    # ROC-AUC and PR-AUC for Soybean (targets == 0 is positive)
    binary_targets_soy = (targets == 0).astype(int)
    try:
        roc_auc = float(roc_auc_score(binary_targets_soy, probs_soy))
    except Exception:
        roc_auc = 0.5
        
    prec_arr, rec_arr, _ = precision_recall_curve(binary_targets_soy, probs_soy)
    pr_auc = float(auc(rec_arr, prec_arr))

    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    # TP: True Soybean (Target 0, Pred 0)
    # FN: Target 0, Pred 1
    # FP: Target 1, Pred 0
    # TN: Target 1, Pred 1
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "soybean_recall": soybean_recall,
        "non_soybean_recall": non_soybean_recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold)
    }

def main():
    device = CONFIG["device"]
    train_ds = CropVerificationDataset("data/splits/crop_train.csv", transform=train_transforms)
    val_ds = CropVerificationDataset("data/splits/crop_val.csv", transform=eval_transforms)
    test_ds = CropVerificationDataset("data/splits/crop_test.csv", transform=eval_transforms)

    # Count dataset classes
    df_meta = pd.read_csv("data/metadata/crop_verification_metadata.csv")
    soy_count = int((df_meta["class_name"] == "Soybean").sum())
    cotton_count = int((df_meta["class_name"] == "Cotton").sum())

    # Pre-training output block as requested
    print("Dataset:", flush=True)
    print(f"Soybean images: {soy_count}", flush=True)
    print(f"Non-Soybean images: {cotton_count}", flush=True)
    print("", flush=True)
    print(f"Train:\n{len(train_ds)}", flush=True)
    print("", flush=True)
    print(f"Validation:\n{len(val_ds)}", flush=True)
    print("", flush=True)
    print(f"Test:\n{len(test_ds)}", flush=True)
    print("", flush=True)
    print("Class mapping:", flush=True)
    print("0 = Soybean", flush=True)
    print("1 = Non_Soybean", flush=True)
    print("", flush=True)
    print(f"Device:\n{device.upper()}", flush=True)
    print("", flush=True)
    print(f"Model:\n{CONFIG['model_name']}", flush=True)
    print("", flush=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Model architecture
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, CONFIG["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    os.makedirs("experiments/crop_verification", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    log_records = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    # ---------------------------------------------------------
    # STAGE 1: CLASSIFICATION HEAD WARMUP
    # ---------------------------------------------------------
    print("--- STAGE 1: Classification Head Warmup (Backbone Frozen) ---", flush=True)
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
        val_metrics = compute_metrics(val_probs, val_targets, threshold=0.5)
        elapsed = time.time() - start_time

        print(f"Stage 1 - Epoch [{epoch}/{CONFIG['stage1_epochs']}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: N/A | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Prec: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}", flush=True)

        log_records.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": CONFIG["stage1_lr"]
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), "models/checkpoints/crop_verification_best.pt")
            torch.save(model.state_dict(), "experiments/crop_verification/best_checkpoint.pt")

    # ---------------------------------------------------------
    # STAGE 2: FULL FINE-TUNING
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
        val_metrics = compute_metrics(val_probs, val_targets, threshold=0.5)
        elapsed = time.time() - start_time

        overall_epoch = CONFIG["stage1_epochs"] + epoch
        print(f"Stage 2 - Epoch [{epoch}/{CONFIG['stage2_epochs']}] ({elapsed:.1f}s) - LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Prec: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}", flush=True)

        log_records.append({
            "stage": 2,
            "epoch": overall_epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": current_lr
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = overall_epoch
            patience_counter = 0
            print(f"  --> Best checkpoint improved! Val F1: {best_val_f1:.4f} (Epoch {best_epoch}). Saving...", flush=True)
            torch.save(model.state_dict(), "models/checkpoints/crop_verification_best.pt")
            torch.save(model.state_dict(), "experiments/crop_verification/best_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement on Val F1.", flush=True)
                break

    pd.DataFrame(log_records).to_csv("experiments/crop_verification/training_log.csv", index=False)

    # ---------------------------------------------------------
    # VALIDATION THRESHOLD TUNING (VALIDATION SET ONLY)
    # ---------------------------------------------------------
    print("\n--- Tuning Probability Threshold on Validation Data Only ---", flush=True)
    best_weights_path = "models/checkpoints/crop_verification_best.pt"
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    val_probs, val_targets = get_probabilities(model, val_loader, device)

    threshold_results = []
    selected_threshold = 0.50
    best_thresh_f1 = -1.0

    print("Candidate Threshold Evaluation (Validation Data):", flush=True)
    for th in CONFIG["candidate_thresholds"]:
        res = compute_metrics(val_probs, val_targets, threshold=th)
        threshold_results.append({
            "threshold": float(th),
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "soybean_recall": res["soybean_recall"],
            "non_soybean_recall": res["non_soybean_recall"]
        })
        print(f"  Th: {th:.2f} | Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
              f"Soy Recall: {res['soybean_recall']:.4f} | Non-Soy Recall: {res['non_soybean_recall']:.4f} | F1: {res['f1']:.4f}", flush=True)
        # Select threshold maximizing Validation F1 with tie-breaker on Soybean recall
        score = res["f1"] * 1000 + res["soybean_recall"]
        if score > best_thresh_f1:
            best_thresh_f1 = score
            selected_threshold = th

    print(f"\n--> Selected Optimal Validation Threshold: {selected_threshold:.2f}", flush=True)

    threshold_payload = {
        "model": CONFIG["model_name"],
        "task": "Soybean_vs_Non_Soybean_Crop_Verification",
        "selected_threshold": float(selected_threshold),
        "validation_metric_used": "Validation F1 with Soybean Recall priority",
        "class_mapping": label_map,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": CONFIG["seed"],
        "candidate_threshold_evaluations": threshold_results
    }
    threshold_path = "models/labels/crop_verification_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_payload, f, indent=2)
    print(f"Threshold saved to {threshold_path}", flush=True)

    # ---------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET
    # ---------------------------------------------------------
    print(f"\n--- Running Single Final Evaluation on Untouched Test Set (Threshold = {selected_threshold:.2f}) ---", flush=True)
    test_probs, test_targets = get_probabilities(model, test_loader, device)
    test_metrics = compute_metrics(test_probs, test_targets, threshold=selected_threshold)

    # Confusion matrix plot
    cm = np.array(test_metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Crop Verification Confusion Matrix (Th={selected_threshold:.2f}, Acc={test_metrics['accuracy']*100:.1f}%)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    cm_path = "experiments/crop_verification/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save metrics.json & config.yaml
    final_results = {
        "model": CONFIG["model_name"],
        "task": CONFIG["task"],
        "best_epoch": best_epoch,
        "selected_threshold": float(selected_threshold),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "soybean_recall": test_metrics["soybean_recall"],
        "non_soybean_recall": test_metrics["non_soybean_recall"],
        "tp": test_metrics["tp"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "confusion_matrix": test_metrics["confusion_matrix"]
    }
    with open("experiments/crop_verification/metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    with open("experiments/crop_verification/config.yaml", "w") as f:
        yaml.dump(CONFIG, f)

    # ---------------------------------------------------------
    # ONNX EXPORT & VERIFICATION
    # ---------------------------------------------------------
    print("\n--- Exporting Model to ONNX ---", flush=True)
    os.makedirs("models/exports", exist_ok=True)
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"])
    onnx_path = "models/exports/crop_verification_efficientnetv2_s.onnx"
    
    export_success = False
    onnx_check_passed = False
    onnx_shape = None

    try:
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
        export_success = True
        print(f"ONNX exported successfully to {onnx_path}", flush=True)

        # Verify with ONNXRuntime
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_shape = ort_outs[0].shape

        # Compare PyTorch and ONNX outputs
        with torch.no_grad():
            py_out = model(dummy_input).numpy()
        max_diff = np.max(np.abs(py_out - ort_outs[0]))
        if onnx_shape == (1, 2) and max_diff < 1e-4:
            onnx_check_passed = True
            print(f"ONNXRuntime Check: PASSED (Shape: {onnx_shape}, max diff: {max_diff:.6e})", flush=True)
        else:
            print(f"ONNXRuntime Check: FAILED (Shape: {onnx_shape}, max diff: {max_diff})", flush=True)
    except Exception as e:
        print(f"ONNX Export/Inference error: {e}", flush=True)

    # ---------------------------------------------------------
    # FINAL RESULTS SUMMARY
    # ---------------------------------------------------------
    print("\n==================================================", flush=True)
    print("CROP VERIFICATION FINAL RESULTS", flush=True)
    print("==================================================", flush=True)
    print(f"Model: {CONFIG['model_name']}", flush=True)
    print("", flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print("", flush=True)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%", flush=True)
    print(f"Test Precision: {test_metrics['precision']:.4f}", flush=True)
    print(f"Test Recall: {test_metrics['recall']:.4f}", flush=True)
    print(f"Test F1: {test_metrics['f1']:.4f}", flush=True)
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}", flush=True)
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}", flush=True)
    print("", flush=True)
    print(f"Soybean Recall: {test_metrics['soybean_recall']*100:.2f}%", flush=True)
    print(f"Non-Soybean Recall: {test_metrics['non_soybean_recall']*100:.2f}%", flush=True)
    print("", flush=True)
    print(f"Selected Validation Threshold: {selected_threshold:.2f}", flush=True)
    print("", flush=True)
    print(f"ONNX Export:\n{'SUCCESS' if export_success else 'FAILED'}", flush=True)
    print("", flush=True)
    print(f"ONNXRuntime Check:\n{'PASSED' if onnx_check_passed else 'FAILED'}", flush=True)
    print("", flush=True)
    print(f"ONNX Output Shape:\n({onnx_shape[0]}, {onnx_shape[1]})" if onnx_shape else "(None, 2)", flush=True)
    print("", flush=True)
    print(f"Best checkpoint:\n{os.path.abspath('models/checkpoints/crop_verification_best.pt')}", flush=True)
    print("", flush=True)
    print(f"ONNX model:\n{os.path.abspath(onnx_path)}", flush=True)
    print("", flush=True)
    print(f"Threshold file:\n{os.path.abspath(threshold_path)}", flush=True)
    print("", flush=True)
    print(f"Label file:\n{os.path.abspath(LABEL_MAP_PATH)}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
