"""
Crop_AI: Extend YOLO11m Training from 20 to 100 Total Epochs
Continues:
1. Dataset A: Soybean Crop Disease (epochs 21 to 100)
2. Dataset B: Nutrient Deficiency (epochs 21 to 100)

Features:
- Initialized directly from verified 20-epoch last.pt weights
- Checkpoints saved every 5 epochs: epoch_025.pt, epoch_030.pt, ..., epoch_100.pt
- Checkpoints 005 to 020 preserved
- Checkpoint logs updated in logs/
- Completely isolated from CNN Stage 8 and SoyCotton models
"""

import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def extend_training(dataset_name, data_yaml_path, model_dir, log_file, initial_epochs=20, add_epochs=80):
    total_target = initial_epochs + add_epochs
    print(f"\n=======================================================")
    print(f"Extending {dataset_name} from Epoch {initial_epochs} to {total_target}...")
    print(f"=======================================================")

    model_dir = Path(model_dir)
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_file)

    # Initialize YOLO from 20-epoch last.pt
    last_pt_path = model_dir / "last.pt"
    if not last_pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {last_pt_path}")

    model = YOLO(str(last_pt_path))

    # Add callback for cumulative 5-epoch checkpointing
    def on_fit_epoch_end_callback(trainer):
        # trainer.epoch is 0-indexed within the new 80-epoch run
        current_sub_epoch = trainer.epoch + 1
        cumulative_epoch = initial_epochs + current_sub_epoch

        metrics_dict = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            metrics_dict = {
                "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(trainer.metrics.get("metrics/precision(B)", 0.0)),
                "recall": float(trainer.metrics.get("metrics/recall(B)", 0.0))
            }
        
        timestamp = datetime.now().isoformat()
        if cumulative_epoch % 5 == 0 or cumulative_epoch == total_target:
            ckpt_name = f"epoch_{cumulative_epoch:03d}.pt"
            ckpt_dest = checkpoints_dir / ckpt_name
            src_last = Path(trainer.last)
            if src_last.exists():
                shutil.copy2(src_last, ckpt_dest)
                print(f"[{dataset_name} Checkpoint] Saved {ckpt_name} (Cumulative Epoch {cumulative_epoch})")
                metrics_str = json.dumps(metrics_dict).replace('"', '""')
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f'{cumulative_epoch},"{ckpt_dest.as_posix()}","completed","{timestamp}","{metrics_str}"\n')

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

    training_config = {
        "model": str(last_pt_path),
        "task": "detect",
        "data": str(data_yaml_path),
        "epochs": add_epochs,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "seed": 42,
        "workers": 8,
        "project": (model_dir / "runs_ext").as_posix(),
        "name": "train_100",
        "exist_ok": True,
        "save_period": 5,
        "verbose": True
    }

    results = model.train(**training_config)

    # Copy best and last to models/yolo11m_.../
    ext_best_src = Path(results.save_dir) / "weights" / "best.pt"
    ext_last_src = Path(results.save_dir) / "weights" / "last.pt"

    if ext_best_src.exists():
        shutil.copy2(ext_best_src, model_dir / "best.pt")
        print(f"Updated best model -> {model_dir / 'best.pt'}")

    if ext_last_src.exists():
        shutil.copy2(ext_last_src, model_dir / "last.pt")
        print(f"Updated last model -> {model_dir / 'last.pt'}")

    summary = {
        "model_architecture": "YOLO11m",
        "dataset": dataset_name,
        "initial_epochs": initial_epochs,
        "extended_epochs": add_epochs,
        "total_epochs": total_target,
        "best_weights": (model_dir / "best.pt").as_posix(),
        "last_weights": (model_dir / "last.pt").as_posix(),
        "total_checkpoints": len(list(checkpoints_dir.glob("epoch_*.pt"))),
        "completed_at": datetime.now().isoformat()
    }
    with open(model_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{dataset_name} reached {total_target} total epochs successfully.")

def main():
    base_dir = Path("c:/Users/oswal/Music/Crop_AI")

    # 1. Dataset A (Disease)
    extend_training(
        dataset_name="Soybean Crop Disease v10",
        data_yaml_path=base_dir / "data" / "yolo_disease" / "data.yaml",
        model_dir=base_dir / "models" / "yolo11m_disease_test",
        log_file=base_dir / "logs" / "yolo11m_disease_test_checkpoint_log.csv",
        initial_epochs=20,
        add_epochs=80
    )

    # 2. Dataset B (Nutrient)
    extend_training(
        dataset_name="Nutrient Deficiency Obj v1",
        data_yaml_path=base_dir / "data" / "yolo_nutrient" / "data.yaml",
        model_dir=base_dir / "models" / "yolo11m_nutrient_test",
        log_file=base_dir / "logs" / "yolo11m_nutrient_test_checkpoint_log.csv",
        initial_epochs=20,
        add_epochs=80
    )

if __name__ == "__main__":
    main()
