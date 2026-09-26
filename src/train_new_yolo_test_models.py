"""
YOLO11m 20-Epoch Test Training Pipeline for:
1. Dataset A: Soybean Crop Disease (models/yolo11m_disease_test/)
2. Dataset B: Nutrient Deficiency (models/yolo11m_nutrient_test/)

Enforces:
- Pretrained yolo11m.pt base weights
- 20 epochs minimum
- batch=16, imgsz=640, seed=42, device=0
- Checkpoints saved every 5 epochs: epoch_005.pt, epoch_010.pt, epoch_015.pt, epoch_020.pt
- Separate logging in logs/
- Completely isolates and protects existing models
"""

import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def train_model(dataset_name, data_yaml_path, output_dir, log_file, target_epochs=20):
    print(f"\n=======================================================")
    print(f"Starting 20-Epoch YOLO11m Test Training: {dataset_name}")
    print(f"=======================================================")

    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("epoch,checkpoint_path,training_status,timestamp,metrics_if_available\n")

    model = YOLO("yolo11m.pt")

    def on_fit_epoch_end_callback(trainer):
        epoch = trainer.epoch + 1
        metrics_dict = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            metrics_dict = {
                "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(trainer.metrics.get("metrics/precision(B)", 0.0)),
                "recall": float(trainer.metrics.get("metrics/recall(B)", 0.0))
            }
        
        timestamp = datetime.now().isoformat()
        if epoch % 5 == 0 or epoch == trainer.epochs:
            ckpt_name = f"epoch_{epoch:03d}.pt"
            ckpt_dest = checkpoints_dir / ckpt_name
            src_last = Path(trainer.last)
            if src_last.exists():
                shutil.copy2(src_last, ckpt_dest)
                print(f"[{dataset_name} Checkpoint] Saved {ckpt_name} at epoch {epoch}")
                metrics_str = json.dumps(metrics_dict).replace('"', '""')
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f'{epoch},"{ckpt_dest.as_posix()}","completed","{timestamp}","{metrics_str}"\n')

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

    training_config = {
        "model": "yolo11m.pt",
        "task": "detect",
        "data": str(data_yaml_path),
        "epochs": target_epochs,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "seed": 42,
        "workers": 8,
        "project": (output_dir / "runs").as_posix(),
        "name": "train",
        "exist_ok": True,
        "save_period": 5,
        "verbose": True
    }

    # Save training config
    with open(output_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    results = model.train(**training_config)

    # Copy best and last
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    last_src = Path(results.save_dir) / "weights" / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, output_dir / "best.pt")
        print(f"Preserved best model -> {output_dir / 'best.pt'}")

    if last_src.exists():
        shutil.copy2(last_src, output_dir / "last.pt")
        print(f"Preserved last model -> {output_dir / 'last.pt'}")

    summary = {
        "model_architecture": "YOLO11m",
        "dataset": dataset_name,
        "epochs_completed": target_epochs,
        "best_weights": (output_dir / "best.pt").as_posix(),
        "last_weights": (output_dir / "last.pt").as_posix(),
        "checkpoints_count": len(list(checkpoints_dir.glob("epoch_*.pt"))),
        "completed_at": datetime.now().isoformat()
    }
    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{dataset_name} training completed successfully.")

def main():
    base_dir = Path("c:/Users/oswal/Music/Crop_AI")
    
    # 1. Train Disease Model
    train_model(
        dataset_name="Soybean Crop Disease v10",
        data_yaml_path=base_dir / "data" / "yolo_disease" / "data.yaml",
        output_dir=base_dir / "models" / "yolo11m_disease_test",
        log_file=base_dir / "logs" / "yolo11m_disease_test_checkpoint_log.csv",
        target_epochs=20
    )

    # 2. Train Nutrient Model
    train_model(
        dataset_name="Nutrient Deficiency Obj v1",
        data_yaml_path=base_dir / "data" / "yolo_nutrient" / "data.yaml",
        output_dir=base_dir / "models" / "yolo11m_nutrient_test",
        log_file=base_dir / "logs" / "yolo11m_nutrient_test_checkpoint_log.csv",
        target_epochs=20
    )

if __name__ == "__main__":
    main()
