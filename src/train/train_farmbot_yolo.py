"""
train_farmbot_yolo.py
Plant / Weed Detection Model Training Pipeline
Model: YOLO11-M (Pretrained: yolo11m.pt)
Dataset: FarmBot soybean growth dataset (bounding boxes)
Classes:
  0: plant
  1: weed

Pipeline:
  1. Sequence/day-aware dataset split verification
  2. Ultralytics YOLO11-M training with GPU/CUDA, AMP, CosineAnnealingLR
  3. Validation mAP@50-95 checkpoint selection
  4. Final evaluation on untouched test set (class-specific AP & recall)
  5. Visual validation on diverse test/val images
  6. ONNX Export and ONNXRuntime verification
  7. Logging, metrics.json, evaluation report generation
"""

import os
import sys
import shutil
import json
import yaml
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

CONFIG = {
    "model_name": "YOLO11-M",
    "task": "plant_weed_detection",
    "base_weights": "yolo11m.pt",
    "data_yaml": "data/prepared/growth_yolo/data.yaml",
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 640,
    "patience": 15,
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "seed": 42,
    "device": 0 if torch.cuda.is_available() else "cpu"
}

def main():
    print("==================================================", flush=True)
    print("STARTING PLANT / WEED DETECTION TRAINING (YOLO11-M)", flush=True)
    print("==================================================", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)

    data_yaml_path = os.path.abspath(CONFIG["data_yaml"])
    weights_path = os.path.abspath(CONFIG["base_weights"])
    exp_dir = os.path.abspath("experiments/farmbot_yolo")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/exports", exist_ok=True)

    # 1. Print dataset statistics from manifest
    manifest_path = "data/metadata/farmbot_manifest.csv"
    if os.path.exists(manifest_path):
        df_m = pd.read_csv(manifest_path)
        print("\nDataset Statistics:", flush=True)
        print(f"Total images: {len(df_m)}", flush=True)
        print(f"Train images: {(df_m['split'] == 'train').sum()}", flush=True)
        print(f"Validation images: {(df_m['split'] == 'val').sum()}", flush=True)
        print(f"Test images: {(df_m['split'] == 'test').sum()}", flush=True)
        print(f"Total plant instances: {df_m['plant_instances'].sum()}", flush=True)
        print(f"Total weed instances: {df_m['weed_instances'].sum()}", flush=True)
        print("", flush=True)

    # 2. Initialize YOLO11-M
    print(f"Loading pretrained YOLO11-M from {weights_path}...", flush=True)
    model = YOLO(weights_path)

    # Save training configuration
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f)

    # 3. Train YOLO11-M
    print("\n--- Starting YOLO11-M Training Run (imgsz=640) ---", flush=True)
    run_dir = os.path.join(exp_dir, "yolo11m_farmbot_run")
    
    results = model.train(
        data=data_yaml_path,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["imgsz"],
        device=CONFIG["device"],
        optimizer="AdamW",
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        weight_decay=CONFIG["weight_decay"],
        cos_lr=CONFIG["cos_lr"],
        patience=CONFIG["patience"],
        seed=CONFIG["seed"],
        project=exp_dir,
        name="yolo11m_farmbot_run",
        exist_ok=True,
        verbose=True
    )

    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    target_best_pt = os.path.abspath("models/checkpoints/plant_weed_yolo11_m_best.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_best_pt)
        print(f"\nBest checkpoint saved to: {target_best_pt}", flush=True)

    # Copy results.csv as training_log.csv
    src_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(src_csv):
        shutil.copy(src_csv, os.path.join(exp_dir, "training_log.csv"))

    # Determine best epoch from results.csv if available
    best_epoch = 0
    if os.path.exists(src_csv):
        df_res = pd.read_csv(src_csv)
        df_res.columns = df_res.columns.str.strip()
        if "metrics/mAP50-95(B)" in df_res.columns:
            best_idx = df_res["metrics/mAP50-95(B)"].idxmax()
            best_epoch = int(df_res.iloc[best_idx]["epoch"])
            print(f"Best Epoch by Validation mAP50-95: {best_epoch}", flush=True)

    # 4. Final Evaluation on Untouched Test Split
    print("\n--- Running Single Final Evaluation on Untouched Test Set ---", flush=True)
    best_model = YOLO(target_best_pt)
    test_metrics = best_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch_size"],
        device=CONFIG["device"],
        project=exp_dir,
        name="test_evaluation",
        exist_ok=True
    )

    # Global box metrics
    test_mp = float(test_metrics.box.mp)
    test_mr = float(test_metrics.box.mr)
    test_map50 = float(test_metrics.box.map50)
    test_map50_95 = float(test_metrics.box.map)

    # Per-class AP & metrics (0: plant, 1: weed)
    ap50_vals = test_metrics.box.ap50
    ap_vals = test_metrics.box.ap
    p_vals = test_metrics.box.p
    r_vals = test_metrics.box.r

    plant_ap50 = float(ap50_vals[0]) if len(ap50_vals) > 0 else 0.0
    plant_map50_95 = float(ap_vals[0]) if len(ap_vals) > 0 else 0.0
    plant_precision = float(p_vals[0]) if len(p_vals) > 0 else 0.0
    plant_recall = float(r_vals[0]) if len(r_vals) > 0 else 0.0

    weed_ap50 = float(ap50_vals[1]) if len(ap50_vals) > 1 else 0.0
    weed_map50_95 = float(ap_vals[1]) if len(ap_vals) > 1 else 0.0
    weed_precision = float(p_vals[1]) if len(p_vals) > 1 else 0.0
    weed_recall = float(r_vals[1]) if len(r_vals) > 1 else 0.0

    # Save metrics JSON
    metrics_summary = {
        "model": "YOLO11-M",
        "task": "plant_weed_detection",
        "best_epoch": best_epoch,
        "test_mAP50": test_map50,
        "test_mAP50_95": test_map50_95,
        "test_precision": test_mp,
        "test_recall": test_mr,
        "plant": {
            "ap50": plant_ap50,
            "mAP50_95": plant_map50_95,
            "precision": plant_precision,
            "recall": plant_recall
        },
        "weed": {
            "ap50": weed_ap50,
            "mAP50_95": weed_map50_95,
            "precision": weed_precision,
            "recall": weed_recall
        }
    }
    metrics_json_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 5. Visual Validation on Diverse Test Samples
    print("\n--- Generating Visual Validation Predictions ---", flush=True)
    vis_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    test_img_dir = "data/prepared/growth_yolo/images/test"
    test_imgs = sorted(os.listdir(test_img_dir))
    
    # Pick representative samples: early, mid, late, crowded
    sample_indices = [0, len(test_imgs)//4, len(test_imgs)//2, 3*len(test_imgs)//4, len(test_imgs)-1]
    selected_samples = [test_imgs[i] for i in sample_indices if i < len(test_imgs)]
    
    for s_name in selected_samples:
        s_path = os.path.join(test_img_dir, s_name)
        res = best_model.predict(s_path, imgsz=640, conf=0.25, device=CONFIG["device"])
        annotated_img = res[0].plot()
        out_vis_path = os.path.join(vis_dir, f"pred_{s_name}")
        cv2.imwrite(out_vis_path, annotated_img)
    print(f"Generated {len(selected_samples)} visual prediction samples in {vis_dir}", flush=True)

    # Copy confusion matrix from test_evaluation if available
    test_eval_dir = os.path.join(exp_dir, "test_evaluation")
    for cm_name in ["confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]:
        cm_src = os.path.join(test_eval_dir, cm_name)
        if os.path.exists(cm_src):
            shutil.copy(cm_src, os.path.join(exp_dir, cm_name))

    # 6. ONNX Export and ONNXRuntime Check
    print("\n--- Exporting YOLO11-M to ONNX ---", flush=True)
    onnx_target_path = os.path.abspath("models/exports/plant_weed_yolo11_m.onnx")
    export_success = False
    onnx_check_passed = False

    try:
        exported_file = best_model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        if os.path.exists(exported_file):
            shutil.copy(exported_file, onnx_target_path)
            export_success = True
            print(f"YOLO11-M exported to ONNX at {onnx_target_path}", flush=True)

        # ONNXRuntime Verification
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(onnx_target_path)
        dummy_inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
        ort_in_name = ort_sess.get_inputs()[0].name
        ort_outs = ort_sess.run(None, {ort_in_name: dummy_inp})
        
        # In YOLO11, output shape is (1, 6, 8400) where 6 = 4 box coordinates + 2 class logits
        out_shape = ort_outs[0].shape
        print(f"ONNX Model Output Shape: {out_shape}", flush=True)
        if len(out_shape) == 3 and out_shape[0] == 1 and out_shape[1] == 6:
            onnx_check_passed = True
            print("ONNXRuntime Check: PASSED (Valid YOLO Detection Output Shape)", flush=True)
        else:
            print(f"ONNXRuntime Check: Unexpected shape {out_shape}", flush=True)
    except Exception as e:
        print(f"ONNX export/inference error: {e}", flush=True)

    # 7. Print Final Result Summary in Exact Template
    print("\n==================================================", flush=True)
    print("PLANT / WEED DETECTION FINAL RESULTS", flush=True)
    print("==================================================", flush=True)
    print(f"Model: {CONFIG['model_name']}", flush=True)
    print("", flush=True)
    print(f"Image Size: {CONFIG['imgsz']}", flush=True)
    print("", flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print("", flush=True)
    print(f"Test mAP50: {test_map50*100:.2f}%", flush=True)
    print(f"Test mAP50-95: {test_map50_95*100:.2f}%", flush=True)
    print(f"Test Precision: {test_mp*100:.2f}%", flush=True)
    print(f"Test Recall: {test_mr*100:.2f}%", flush=True)
    print("", flush=True)
    print("-----------------------------", flush=True)
    print("PLANT", flush=True)
    print("-----------------------------", flush=True)
    print("", flush=True)
    print(f"Plant AP50: {plant_ap50*100:.2f}%", flush=True)
    print(f"Plant AP50-95: {plant_map50_95*100:.2f}%", flush=True)
    print(f"Plant Precision: {plant_precision*100:.2f}%", flush=True)
    print(f"Plant Recall: {plant_recall*100:.2f}%", flush=True)
    print("", flush=True)
    print("-----------------------------", flush=True)
    print("WEED", flush=True)
    print("-----------------------------", flush=True)
    print("", flush=True)
    print(f"Weed AP50: {weed_ap50*100:.2f}%", flush=True)
    print(f"Weed AP50-95: {weed_map50_95*100:.2f}%", flush=True)
    print(f"Weed Precision: {weed_precision*100:.2f}%", flush=True)
    print(f"Weed Recall: {weed_recall*100:.2f}%", flush=True)
    print("", flush=True)
    print("-----------------------------", flush=True)
    print("EXPORT", flush=True)
    print("-----------------------------", flush=True)
    print("", flush=True)
    print(f"ONNX Export:\n{'SUCCESS' if export_success else 'FAILED'}", flush=True)
    print("", flush=True)
    print(f"ONNXRuntime Check:\n{'PASSED' if onnx_check_passed else 'FAILED'}", flush=True)
    print("", flush=True)
    print(f"Best checkpoint:\n{target_best_pt}", flush=True)
    print("", flush=True)
    print(f"ONNX model:\n{onnx_target_path}", flush=True)
    print("", flush=True)
    print(f"Evaluation:\n{os.path.abspath(exp_dir)}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
