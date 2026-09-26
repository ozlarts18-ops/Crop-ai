import os
import json
import csv
import random
import shutil
from collections import defaultdict
from src.config import (
    PROCESSED_DIR, CNN_DIR, METADATA_DIR, OUTPUTS_DIR,
    PLANT_NAME, SCIENTIFIC_NAME, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, RANDOM_SEED
)

def run_stage5():
    print("Executing Stage 5: Split Data (Stratified 70/15/15 Leakage-Free Split)...")
    
    clean_records_path = os.path.join(PROCESSED_DIR, "clean_records.json")
    with open(clean_records_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Set fixed random seed
    random.seed(RANDOM_SEED)
    
    # Group records by normalized class for stratified split
    by_class = defaultdict(list)
    for r in records:
        by_class[r["norm_class"]].append(r)
        
    train_records = []
    val_records = []
    test_records = []
    
    for cls_name, cls_list in sorted(by_class.items()):
        # Shuffle deterministically
        random.shuffle(cls_list)
        n = len(cls_list)
        n_train = int(round(n * SPLIT_TRAIN))
        n_val = int(round(n * SPLIT_VAL))
        # Ensure at least 1 in val and test if n >= 3
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        n_test = n - n_train - n_val
        
        train_part = cls_list[:n_train]
        val_part = cls_list[n_train:n_train + n_val]
        test_part = cls_list[n_train + n_val:]
        
        for item in train_part:
            item["split"] = "train"
        for item in val_part:
            item["split"] = "val"
        for item in test_part:
            item["split"] = "test"
            
        train_records.extend(train_part)
        val_records.extend(val_part)
        test_records.extend(test_part)

    print(f"Total split: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    # Copy / organize CNN folder structures: data/processed/cnn/{split}/{norm_class}/
    for r in train_records + val_records + test_records:
        split = r["split"]
        cls_name = r["norm_class"]
        dest_folder = os.path.join(CNN_DIR, split, cls_name)
        os.makedirs(dest_folder, exist_ok=True)
        dest_file = os.path.join(dest_folder, r["processed_filename"])
        if not os.path.exists(dest_file):
            shutil.copy2(r["processed_path"], dest_file)
        r["cnn_path"] = dest_file

    # Build master_metadata.csv
    master_metadata_path = os.path.join(METADATA_DIR, "master_metadata.csv")
    csv_fields = [
        "image_id", "image_path", "source_dataset", "plant_name", "scientific_name",
        "original_class", "normalized_class", "health_status", "growth_stage",
        "image_type", "location", "season", "split", "has_bbox", "has_segmentation",
        "mask_path", "bbox_path", "leaf_area_pixels", "disease_area_pixels",
        "disease_spread_percent", "severity_class"
    ]
    
    metadata_rows = []
    all_assigned = train_records + val_records + test_records
    for idx, r in enumerate(all_assigned):
        norm_class = r["norm_class"]
        health_status = "Healthy" if norm_class == "Healthy" else "Diseased"
        
        row = {
            "image_id": f"IMG_{idx+1:06d}",
            "image_path": r["cnn_path"].replace("\\", "/"),
            "source_dataset": r["source_dataset"],
            "plant_name": PLANT_NAME,
            "scientific_name": SCIENTIFIC_NAME,
            "original_class": r["raw_class"],
            "normalized_class": norm_class,
            "health_status": health_status,
            "growth_stage": "Vegetative / Reproductive",
            "image_type": r.get("image_type", "leaf"),
            "location": "India",
            "season": "Kharif",
            "split": r["split"],
            "has_bbox": False,
            "has_segmentation": False,
            "mask_path": "null",
            "bbox_path": "null",
            "leaf_area_pixels": "null",
            "disease_area_pixels": "null",
            "disease_spread_percent": "null",
            "severity_class": "null"
        }
        metadata_rows.append(row)

    with open(master_metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Generate outputs/split_report.json
    class_counts_by_split = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for r in all_assigned:
        c = r["norm_class"]
        s = r["split"]
        class_counts_by_split[c][s] += 1
        class_counts_by_split[c]["total"] += 1

    split_report = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": SPLIT_TRAIN,
            "val": SPLIT_VAL,
            "test": SPLIT_TEST
        },
        "total_images": len(all_assigned),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "per_class_split": dict(class_counts_by_split)
    }

    with open(os.path.join(OUTPUTS_DIR, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=4)

    # Generate outputs/class_distribution.csv
    dist_csv_path = os.path.join(OUTPUTS_DIR, "class_distribution.csv")
    with open(dist_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_class", "train", "val", "test", "total"])
        for cls_name, counts in sorted(class_counts_by_split.items()):
            writer.writerow([cls_name, counts["train"], counts["val"], counts["test"], counts["total"]])

    # Generate outputs/class_distribution.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        classes = sorted(list(class_counts_by_split.keys()))
        train_counts = [class_counts_by_split[c]["train"] for c in classes]
        val_counts = [class_counts_by_split[c]["val"] for c in classes]
        test_counts = [class_counts_by_split[c]["test"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, train_counts, width, label='Train', color='#2ecc71')
        ax.bar(x, val_counts, width, label='Val', color='#f39c12')
        ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c')

        ax.set_ylabel('Image Count')
        ax.set_title('Soybean Dataset Split Distribution by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "class_distribution.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot with matplotlib: {e}")

    print(f"Stage 5 complete: master_metadata.csv, split_report.json, and class_distribution created.")
    return split_report

if __name__ == "__main__":
    run_stage5()
