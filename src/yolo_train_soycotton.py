"""
Crop_AI: YOLO11m Object Detection Training Pipeline on SoyCotton
- Architecture: YOLO11m (Detect)
- Dataset: data/yolo_soycotton/data.yaml (Soybean leaf annotations ONLY)
- Epochs: 100
- Imgsz: 640
- Batch: 16 (on NVIDIA RTX 4070 12GB)
- Seed: 42
- Checkpointing:
  * Every 5 epochs produces epoch_005.pt, epoch_010.pt, ..., epoch_100.pt
  * Checkpoints stored in models/yolo11m/checkpoints/
  * Logged to logs/yolo11m_checkpoint_log.csv
  * Preserves models/yolo11m/best.pt and models/yolo11m/last.pt
  * Generates models/yolo11m/training_config.yaml and model_summary.json
  * Strict separation: DOES NOT touch models/optimized/best_model.pt
"""

import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path("c:/Users/oswal/Music/Crop_AI")
    data_yaml = base_dir / "data" / "yolo_soycotton" / "data.yaml"
    
    models_dir = base_dir / "models" / "yolo11m"
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log_file = logs_dir / "yolo11m_checkpoint_log.csv"

    # Initialize checkpoint log if not exists
    if not checkpoint_log_file.exists():
        with open(checkpoint_log_file, "w", encoding="utf-8") as f:
            f.write("epoch,checkpoint_path,training_status,timestamp,metrics_if_available\n")

    print(f"Initializing YOLO11m training on device=0 (NVIDIA RTX 4070)...")
    model = YOLO("yolo11m.pt")

    # Define callback to save checkpoints every 5 epochs
    def on_fit_epoch_end_callback(trainer):
        epoch = trainer.epoch + 1
        metrics_dict = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            # Extract main metrics
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
            
            # Copy latest weights to checkpoint destination
            src_last = Path(trainer.last)
            if src_last.exists():
                shutil.copy2(src_last, ckpt_dest)
                print(f"[Checkpoint] Saved {ckpt_name} at epoch {epoch}")
                
                metrics_str = json.dumps(metrics_dict).replace('"', '""')
                with open(checkpoint_log_file, "a", encoding="utf-8") as f:
                    f.write(f'{epoch},"{ckpt_dest.as_posix()}","completed","{timestamp}","{metrics_str}"\n')

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

    training_config = {
        "model": "yolo11m.pt",
        "task": "detect",
        "data": data_yaml.as_posix(),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "seed": 42,
        "workers": 8,
        "project": (base_dir / "outputs" / "yolo11m_runs").as_posix(),
        "name": "train",
        "exist_ok": True,
        "save_period": 5,
        "verbose": True
    }

    # Save training config to models/yolo11m/training_config.yaml
    with open(models_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    print("Starting 100 epoch training run...")
    results = model.train(**training_config)

    # Copy best.pt and last.pt to models/yolo11m/
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    last_src = Path(results.save_dir) / "weights" / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, models_dir / "best.pt")
        print(f"Preserved best model -> {models_dir / 'best.pt'}")

    if last_src.exists():
        shutil.copy2(last_src, models_dir / "last.pt")
        print(f"Preserved last model -> {models_dir / 'last.pt'}")

    # Generate model_summary.json
    summary = {
        "model_architecture": "YOLO11m",
        "task": "detect",
        "classes": {0: "soybean_leaf"},
        "epochs_completed": 100,
        "training_time_hours": getattr(results, "train_time", None),
        "best_weights_path": (models_dir / "best.pt").as_posix(),
        "last_weights_path": (models_dir / "last.pt").as_posix(),
        "checkpoints_directory": checkpoints_dir.as_posix(),
        "total_checkpoints_created": len(list(checkpoints_dir.glob("epoch_*.pt"))),
        "training_completed_at": datetime.now().isoformat()
    }
    with open(models_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("YOLO11m training run and artifact preservation completed successfully.")

if __name__ == "__main__":
    main()
