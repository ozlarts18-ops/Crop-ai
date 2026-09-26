import os
import json
import csv
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CNN_DIR, MODELS_DIR, OUTPUTS_DIR, NORMALIZED_CLASSES
from src.stage6_train_cnn import build_model

def evaluate_cnn():
    print("="*60)
    print("STAGE 7: EVALUATING CNN ON TEST SET")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(MODELS_DIR, "cnn", "best.pt")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best CNN model checkpoint not found at {best_model_path}")
        
    ckpt = torch.load(best_model_path, map_location=device)
    class_names = ckpt["class_names"]
    
    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    test_dir = os.path.join(CNN_DIR, "test")
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    clf_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    clf_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    cnn_eval_dir = os.path.join(OUTPUTS_DIR, "evaluation", "cnn")
    os.makedirs(cnn_eval_dir, exist_ok=True)
    
    # Save classification report txt & json
    with open(os.path.join(cnn_eval_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(clf_text)
        
    full_report = {
        "overall": {
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "precision_weighted": p_weighted,
            "recall_weighted": r_weighted,
            "f1_weighted": f1_weighted,
            "total_test_samples": len(all_labels)
        },
        "per_class": clf_dict
    }
    with open(os.path.join(cnn_eval_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)
        
    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='CNN Confusion Matrix (Test Set)',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(cnn_eval_dir, "confusion_matrix.png"), dpi=200)
    plt.close()
    
    print("CNN Evaluation Complete. Saved classification report and confusion matrix.")
    return full_report, cm

def evaluate_yolo():
    print("STAGE 7: EVALUATING YOLO...")
    yolo_eval_dir = os.path.join(OUTPUTS_DIR, "evaluation", "yolo")
    os.makedirs(yolo_eval_dir, exist_ok=True)
    
    # Check if localization existed
    yolo_metrics = {
        "status": "Unavailable",
        "reason": "Source datasets contain 0 bounding-box localization annotations. Strict compliance with prompt directive: fake bounding boxes were not created.",
        "Precision": None,
        "Recall": None,
        "F1": None,
        "mAP50": None,
        "mAP50-95": None,
        "IoU": None
    }
    with open(os.path.join(yolo_eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(yolo_metrics, f, indent=4)
        
    with open(os.path.join(yolo_eval_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "mAP50", "mAP50-95", "status"])
        writer.writerow(["All", "null", "null", "null", "null", "null", "No localization annotations in raw datasets"])
        
    # Generate placeholder/explanatory plots
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "YOLO Detection: No Bounding Boxes in Source Data\n(Fake annotations avoided per prompt instructions)",
            ha='center', va='center', fontsize=11, color='darkred')
    ax.axis('off')
    plt.savefig(os.path.join(yolo_eval_dir, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(yolo_eval_dir, "results.png"), dpi=150)
    plt.close()
    
    print("YOLO evaluation artifact recorded.")
    return yolo_metrics

def evaluate_spreadness():
    print("STAGE 7: EVALUATING SPREADNESS...")
    spread_dir = os.path.join(OUTPUTS_DIR, "evaluation", "spread")
    os.makedirs(spread_dir, exist_ok=True)
    
    spread_report = {
        "status": "Segmentation unavailable",
        "disease_spread_percent": None,
        "disease_area_pixels": None,
        "leaf_area_pixels": None,
        "notes": "Valid segmentation masks are absent from source datasets. Under strict prompt rules ('DO NOT use CNN confidence as spreadness', 'DO NOT invent spread percentages', 'If valid segmentation information does not exist: set disease_spread_percent = null and report: Segmentation unavailable'), spreadness is formally logged as null."
    }
    
    with open(os.path.join(spread_dir, "spread_report.json"), "w", encoding="utf-8") as f:
        json.dump(spread_report, f, indent=4)
        
    with open(os.path.join(spread_dir, "spread_report.txt"), "w", encoding="utf-8") as f:
        f.write("SPREADNESS ANALYSIS REPORT\n")
        f.write("==========================\n")
        f.write("Status: Segmentation unavailable\n")
        f.write("disease_spread_percent: null\n")
        f.write("disease_area_pixels: null\n")
        f.write("leaf_area_pixels: null\n\n")
        f.write(spread_report["notes"] + "\n")
        
    print("Spreadness report written to outputs/evaluation/spread/")
    return spread_report

def generate_final_report(cnn_report, yolo_metrics, spread_report):
    print("="*60)
    print("GENERATING FINAL TEST REPORT")
    print("="*60)
    
    # Read data collection & split reports
    with open(os.path.join(OUTPUTS_DIR, "data_collection_report.json"), "r", encoding="utf-8") as f:
        col_rep = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "split_report.json"), "r", encoding="utf-8") as f:
        spl_rep = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, "data_quality_report.json"), "r", encoding="utf-8") as f:
        qual_rep = json.load(f)
        
    cnn_checkpoints = sorted(glob.glob(os.path.join(MODELS_DIR, "cnn", "checkpoints", "*.pt")))
    
    final_data = {
        "project_title": "Crop_AI Soybean Crop Health & Disease Detection",
        "plant_name": "Soybean",
        "scientific_name": "Glycine max",
        "datasets_processed": len(col_rep),
        "total_images_scanned": qual_rep["total_images_scanned"],
        "clean_unique_images": qual_rep["clean_unique_images"],
        "duplicates_detected": qual_rep["duplicate_images_count"],
        "corrupted_images_detected": qual_rep["corrupted_images_count"],
        "unknown_classes_detected": qual_rep["unknown_classes_count"],
        "splits": {
            "training_images": spl_rep["split_counts"]["train"],
            "validation_images": spl_rep["split_counts"]["val"],
            "test_images": spl_rep["split_counts"]["test"]
        },
        "model_status": {
            "cnn_training_status": "Completed 100 Epochs (EfficientNet-B0)",
            "yolo_training_status": yolo_metrics["status"]
        },
        "model_locations": {
            "cnn_best_checkpoint": os.path.join(MODELS_DIR, "cnn", "best.pt"),
            "cnn_last_checkpoint": os.path.join(MODELS_DIR, "cnn", "last.pt"),
            "cnn_5epoch_checkpoints": cnn_checkpoints,
            "yolo_best_checkpoint": "N/A (No localization data in source)"
        },
        "cnn_test_metrics": cnn_report["overall"],
        "cnn_per_class_metrics": cnn_report["per_class"],
        "yolo_test_metrics": yolo_metrics,
        "spreadness_evaluation": spread_report
    }
    
    # Save final_test_report.json
    final_json_path = os.path.join(OUTPUTS_DIR, "final_test_report.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)
        
    # Save final_test_report.txt
    final_txt_path = os.path.join(OUTPUTS_DIR, "final_test_report.txt")
    with open(final_txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("                        CROP_AI FINAL TEST REPORT (STAGES 1 - 7)\n")
        f.write("================================================================================\n\n")
        f.write(f"Target Plant: {final_data['plant_name']} ({final_data['scientific_name']})\n")
        f.write(f"Total Datasets Processed: {final_data['datasets_processed']}\n")
        f.write(f"Total Images Scanned: {final_data['total_images_scanned']}\n")
        f.write(f"Clean Unique Images: {final_data['clean_unique_images']}\n")
        f.write(f"Training Images: {final_data['splits']['training_images']}\n")
        f.write(f"Validation Images: {final_data['splits']['validation_images']}\n")
        f.write(f"Test Images: {final_data['splits']['test_images']}\n\n")
        
        f.write("--- MODEL TRAINING STATUS ---\n")
        f.write(f"CNN Status: {final_data['model_status']['cnn_training_status']}\n")
        f.write(f"YOLO Status: {final_data['model_status']['yolo_training_status']}\n\n")
        
        f.write("--- CHECKPOINT LOCATIONS ---\n")
        f.write(f"CNN Best Checkpoint: {final_data['model_locations']['cnn_best_checkpoint']}\n")
        f.write(f"CNN Last Checkpoint: {final_data['model_locations']['cnn_last_checkpoint']}\n")
        f.write(f"CNN 5-Epoch Checkpoints Count: {len(cnn_checkpoints)}\n")
        for cp in cnn_checkpoints:
            f.write(f"  - {cp}\n")
        f.write(f"YOLO Best Checkpoint: {final_data['model_locations']['yolo_best_checkpoint']}\n\n")
        
        f.write("--- CNN TEST SET PERFORMANCE (EFFICIENTNET-B0) ---\n")
        f.write(f"Accuracy: {final_data['cnn_test_metrics']['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {final_data['cnn_test_metrics']['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {final_data['cnn_test_metrics']['recall_macro']:.4f}\n")
        f.write(f"Macro F1-Score: {final_data['cnn_test_metrics']['f1_macro']:.4f}\n")
        f.write(f"Weighted Precision: {final_data['cnn_test_metrics']['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall: {final_data['cnn_test_metrics']['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score: {final_data['cnn_test_metrics']['f1_weighted']:.4f}\n\n")
        
        f.write("--- YOLO TEST SET PERFORMANCE ---\n")
        f.write(f"Status: {final_data['yolo_test_metrics']['status']}\n")
        f.write(f"Reason: {final_data['yolo_test_metrics']['reason']}\n\n")
        
        f.write("--- SPREADNESS EVALUATION ---\n")
        f.write(f"Status: {final_data['spreadness_evaluation']['status']}\n")
        f.write(f"Spread Percent: {final_data['spreadness_evaluation']['disease_spread_percent']}\n")
        f.write(f"Notes: {final_data['spreadness_evaluation']['notes']}\n\n")
        
        f.write("--- DATASET PROBLEMS & EXCLUSIONS ---\n")
        f.write(f"Corrupted Images: {final_data['corrupted_images_detected']} (logged in outputs/corrupted_images.csv)\n")
        f.write(f"Duplicates/Near-Duplicates: {final_data['duplicates_detected']} (logged in outputs/duplicates.csv)\n")
        f.write(f"Unknown Classes: {final_data['unknown_classes_detected']} (logged in outputs/unknown_classes.csv)\n")
        f.write("Excluded Classes: Any images with non-specific labels (such as 'Disease' or 'Mobile pic' in SoyNet) were excluded from multi-class training and logged as UNKNOWN to prevent label corruption.\n")
        
    print(f"Final test report successfully generated: {final_json_path} and {final_txt_path}")
    return final_data

def run_stage7():
    cnn_report, cm = evaluate_cnn()
    yolo_metrics = evaluate_yolo()
    spread_report = evaluate_spreadness()
    final_data = generate_final_report(cnn_report, yolo_metrics, spread_report)
    return final_data

if __name__ == "__main__":
    run_stage7()
