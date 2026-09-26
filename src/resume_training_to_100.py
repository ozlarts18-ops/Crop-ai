"""
Crop_AI: Resume YOLO11m Training to 100 Total Epochs
Resumes:
1. Dataset A: Soybean Crop Disease (from epoch 20 to 100)
2. Dataset B: Nutrient Deficiency (from epoch 20 to 100)

Enforces:
- Resume from verified last.pt checkpoints
- Checkpoints saved every 5 epochs: epoch_025.pt, epoch_030.pt, ..., epoch_100.pt
- Preserves epoch_005.pt through epoch_020.pt
- Logs tracked in logs/
- Completely isolated from CNN Stage 8 and SoyCotton models
"""

import os
import sys
import json
import yaml
import shutil
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def prepare_checkpoint_for_resume(run_dir, target_epochs=100):
    run_dir = Path(run_dir)
    args_p = run_dir / "args.yaml"
    last_pt_p = run_dir / "weights" / "last.pt"

    if not last_pt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {last_pt_p}")

    # Load and update checkpoint epoch & train_args
    ckpt = torch.load(last_pt_p, map_location="cpu", weights_only=False)
    # If epoch is -1 (from previous finish), set to 19 (meaning 20 epochs completed)
    if ckpt.get("epoch", -1) < 0:
        ckpt["epoch"] = 19
    if "train_args" in ckpt and isinstance(ckpt["train_args"], dict):
        ckpt["train_args"]["epochs"] = target_epochs
    torch.save(ckpt, last_pt_p)

    # Update args.yaml
    if args_p.exists():
        with open(args_p, "r", encoding="utf-8") as yf:
            args_data = yaml.safe_load(yf)
        args_data["epochs"] = target_epochs
        with open(args_p, "w", encoding="utf-8") as yf:
            yaml.dump(args_data, yf)

    print(f"Prepared {last_pt_p} and {args_p} for resume to {target_epochs} total epochs.")
    return last_pt_p

def resume_model_training(dataset_name, model_dir, log_file, target_epochs=100):
    model_dir = Path(model_dir)
    run_dir = model_dir / "runs" / "train"
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_file)

    print(f"\n=======================================================")
    print(f"Resuming YOLO11m Training to {target_epochs} Epochs: {dataset_name}")
    print(f"=======================================================")

    last_ckpt_p = prepare_checkpoint_for_resume(run_dir, target_epochs=target_epochs)

    # Load model from checkpoint
    model = YOLO(str(last_ckpt_p))

    # Add callback for 5-epoch checkpointing
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

    # Resume training
    print(f"Initiating Ultralytics resume for {dataset_name}...")
    results = model.train(resume=True)

    # Copy best and last to models/yolo11m_.../
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"

    if best_src.exists():
        shutil.copy2(best_src, model_dir / "best.pt")
        print(f"Updated best model -> {model_dir / 'best.pt'}")

    if last_src.exists():
        shutil.copy2(last_src, model_dir / "last.pt")
        print(f"Updated last model -> {model_dir / 'last.pt'}")

    summary_file = model_dir / "model_summary.json"
    summary_data = {
        "model_architecture": "YOLO11m",
        "dataset": dataset_name,
        "epochs_completed": target_epochs,
        "best_weights": (model_dir / "best.pt").as_posix(),
        "last_weights": (model_dir / "last.pt").as_posix(),
        "total_checkpoints": len(list(checkpoints_dir.glob("epoch_*.pt"))),
        "completed_at": datetime.now().isoformat()
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"{dataset_name} reached {target_epochs} total epochs successfully.")

def main():
    base_dir = Path("c:/Users/oswal/Music/Crop_AI")
    
    # 1. Resume Disease Model (Dataset A)
    resume_model_training(
        dataset_name="Soybean Crop Disease v10",
        model_dir=base_dir / "models" / "yolo11m_disease_test",
        log_file=base_dir / "logs" / "yolo11m_disease_test_checkpoint_log.csv",
        target_epochs=100
    )

    # 2. Resume Nutrient Model (Dataset B)
    resume_model_training(
        dataset_name="Nutrient Deficiency Obj v1",
        model_dir=base_dir / "models" / "yolo11m_nutrient_test",
        log_file=base_dir / "logs" / "yolo11m_nutrient_test_checkpoint_log.csv",
        target_epochs=100
    )

if __name__ == "__main__":
    main()
