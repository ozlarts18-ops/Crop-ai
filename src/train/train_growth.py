"""
train_growth.py
Training pipeline for Model 3: Growth & Weed Detection on FarmBot Dataset.
Architecture: Lightweight YOLO11n (Ultralytics)
Task: plant vs weed detection and counting
Dataset: data/prepared/growth_yolo/data.yaml (Deduplicated, zero-leakage split across 20 observation days)
Strictly avoids fabricating formal vegetative stages (VE, V1, R1).
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
DATA_DIR = ROOT_DIR / "data"
DATA_YAML = DATA_DIR / "prepared" / "growth_yolo" / "data.yaml"
RUNS_DIR = ROOT_DIR / "runs" / "growth_v1"
MODEL_DIR = ROOT_DIR / "models" / "growth"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test for 1 epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    epochs = 1 if args.smoke else args.epochs

    print("=" * 60)
    print("MODEL 3 — GROWTH / WEED DETECTION TRAINING (YOLO11n)")
    print("=" * 60)
    print(f"Dataset YAML: {DATA_YAML}")
    print("Task: Real Plant vs Weed detection & counting from FarmBot chronological observations")
    print("Classes: ['plant', 'weed']")
    print("NOTE: VE, V1, R1 stages are NOT in dataset and will NOT be claimed.")

    # Load lightweight YOLO11n
    model = YOLO("yolo11n.pt")

    start_time = time.time()
    # Train model
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=str(RUNS_DIR),
        name="train",
        seed=42,
        device=0, # GPU RTX 4070
        exist_ok=True,
        verbose=True
    )
    training_time = time.time() - start_time

    # Validate on pristine test set
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SPLIT")
    print("=" * 60)
    val_results = model.val(data=str(DATA_YAML), split="test", device=0)

    # Extract metrics
    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
        "class_metrics": {}
    }

    # Per-class metrics
    class_names = ['plant', 'weed']
    for i, cname in enumerate(class_names):
        try:
            p_val = float(val_results.box.p[i]) if hasattr(val_results.box, 'p') and len(val_results.box.p) > i else 0.0
            r_val = float(val_results.box.r[i]) if hasattr(val_results.box, 'r') and len(val_results.box.r) > i else 0.0
            map50_val = float(val_results.box.ap50[i]) if hasattr(val_results.box, 'ap50') and len(val_results.box.ap50) > i else 0.0
            metrics["class_metrics"][cname] = {
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "mAP50": round(map50_val, 4)
            }
        except Exception as e:
            metrics["class_metrics"][cname] = {"note": str(e)}

    print(f"Test mAP50:     {metrics['mAP50']}")
    print(f"Test mAP50-95:  {metrics['mAP50-95']}")
    print(f"Test Precision: {metrics['precision']}")
    print(f"Test Recall:    {metrics['recall']}")
    print("Per-class metrics:", metrics["class_metrics"])

    # Save best checkpoint to models/growth/best.pt
    best_train_weight = RUNS_DIR / "train" / "weights" / "best.pt"
    if not best_train_weight.exists():
        best_train_weight = RUNS_DIR / "train" / "weights" / "last.pt"

    dest_best = MODEL_DIR / "best.pt"
    if best_train_weight.exists():
        shutil.copy2(best_train_weight, dest_best)
        print(f"Saved best checkpoint to: {dest_best} ({dest_best.stat().st_size/(1024*1024):.2f} MB)")

    # Save experiment summary
    experiment_summary = {
        "task": "growth_and_weed_detection",
        "model_architecture": "YOLO11n",
        "classes": class_names,
        "dataset": "FarmBot_Soybean_and_Weed",
        "observations": "20_days_chronological",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "seed": 42
        },
        "training_time_seconds": round(training_time, 2),
        "test_metrics": metrics
    }

    with open(RUNS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)

    print("\nMODEL 3 GROWTH / WEED TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
