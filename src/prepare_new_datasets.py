"""
Prepare Isolated YOLO Datasets and Visual Validation
Dataset A: data/yolo_disease/
Dataset B: data/yolo_nutrient/
Visual Validations:
- outputs/new_dataset_audit/disease_visual_validation/
- outputs/new_dataset_audit/nutrient_visual_validation/
"""

import os
import sys
import json
import yaml
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

def prepare_and_validate():
    base_dir = Path("c:/Users/oswal/Music/Crop_AI")
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "outputs" / "new_dataset_audit"
    
    disease_vis_dir = out_dir / "disease_visual_validation"
    nutrient_vis_dir = out_dir / "nutrient_visual_validation"
    disease_vis_dir.mkdir(parents=True, exist_ok=True)
    nutrient_vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Paths
    disease_src = list(raw_dir.glob("*Soybean Crop Disease*"))[0]
    nutrient_src = list(raw_dir.glob("*Nutrient Deficiency*"))[0]

    yolo_disease_dir = base_dir / "data" / "yolo_disease"
    yolo_nutrient_dir = base_dir / "data" / "yolo_nutrient"

    for d in [yolo_disease_dir, yolo_nutrient_dir]:
        for s in ["train", "val", "test"]:
            (d / "images" / s).mkdir(parents=True, exist_ok=True)
            (d / "labels" / s).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # DATASET A: SOYBEAN CROP DISEASE
    # ----------------------------------------------------
    print("Processing Dataset A (Soybean Crop Disease)...")
    with open(disease_src / "data.yaml", "r", encoding="utf-8") as yf:
        disease_yaml = yaml.safe_load(yf)
    disease_classes = disease_yaml["names"]

    # Visual Validation for Disease
    # Map split names from source ('valid' -> 'val')
    src_splits_disease = {"train": "train", "valid": "val", "test": "test"}
    disease_samples_per_class = defaultdict(list)

    # Build isolated dataset
    split_counts_disease = defaultdict(lambda: {"images": 0, "valid_boxes": 0, "excluded_boxes": 0})

    for src_split, dst_split in src_splits_disease.items():
        src_img_dir = disease_src / src_split / "images"
        src_lbl_dir = disease_src / src_split / "labels"

        for img_p in sorted(src_img_dir.glob("*.*")):
            if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue

            lbl_p = src_lbl_dir / (img_p.stem + ".txt")
            valid_lines = []
            img_classes = set()

            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                xc, yc, bw, bh = map(float, parts[1:5])
                                if bw > 0 and bh > 0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0 <= cls_id < len(disease_classes):
                                    valid_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                                    img_classes.add(cls_id)
                                    split_counts_disease[dst_split]["valid_boxes"] += 1
                                else:
                                    split_counts_disease[dst_split]["excluded_boxes"] += 1
                            except ValueError:
                                split_counts_disease[dst_split]["excluded_boxes"] += 1

            # Copy image and label
            dst_img_p = yolo_disease_dir / "images" / dst_split / img_p.name
            dst_lbl_p = yolo_disease_dir / "labels" / dst_split / (img_p.stem + ".txt")
            
            shutil.copy2(img_p, dst_img_p)
            with open(dst_lbl_p, "w", encoding="utf-8") as df:
                df.write("\n".join(valid_lines) + "\n" if valid_lines else "")
            
            split_counts_disease[dst_split]["images"] += 1

            # Visual sample candidate
            for c in img_classes:
                if len(disease_samples_per_class[c]) < 5:
                    disease_samples_per_class[c].append((img_p, valid_lines))

    # Write data/yolo_disease/data.yaml
    disease_data_yaml_content = {
        "path": yolo_disease_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(disease_classes),
        "names": disease_classes
    }
    with open(yolo_disease_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yaml.dump(disease_data_yaml_content, yf, default_flow_style=False)

    # Render Visual Samples for Disease
    print("Rendering Visual Validation for Dataset A...")
    for c_id, sample_list in disease_samples_per_class.items():
        c_name = disease_classes[c_id].replace(" ", "_")
        for idx, (img_p, lines) in enumerate(sample_list):
            im = Image.open(img_p).convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size

            for l in lines:
                cid, xc, yc, bw, bh = l.split()
                cid = int(cid)
                xc, yc, bw, bh = map(float, [xc, yc, bw, bh])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                color = "#FF3333" if cid == c_id else "#FFCC00"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 3, max(0, y1 - 12)), f"{disease_classes[cid]}", fill=color)

            out_fn = f"disease_class_{c_id}_{c_name}_sample_{idx+1}.png"
            im.save(disease_vis_dir / out_fn)

    # ----------------------------------------------------
    # DATASET B: NUTRIENT DEFICIENCY
    # ----------------------------------------------------
    print("Processing Dataset B (Nutrient Deficiency)...")
    with open(nutrient_src / "data.yaml", "r", encoding="utf-8") as yf:
        nutrient_yaml = yaml.safe_load(yf)
    nutrient_classes = nutrient_yaml["names"]

    split_counts_nutrient = defaultdict(lambda: {"images": 0, "valid_boxes": 0, "excluded_boxes": 0})
    nutrient_samples_per_class = defaultdict(list)

    # Source has train (854 images) and valid (211 images)
    # Train goes directly to train
    # Valid is deterministically split (seed 42) into val (105) and test (106)
    valid_imgs = sorted(list((nutrient_src / "valid" / "images").glob("*.*")))
    random.seed(42)
    random.shuffle(valid_imgs)
    val_set = set(valid_imgs[:105])
    test_set = set(valid_imgs[105:])

    print(f"Dataset B partition: Train={len(list((nutrient_src / 'train' / 'images').glob('*.*')))}, Val={len(val_set)}, Test={len(test_set)}")

    # Process train
    for img_p in sorted((nutrient_src / "train" / "images").glob("*.*")):
        lbl_p = nutrient_src / "train" / "labels" / (img_p.stem + ".txt")
        valid_lines = []
        img_classes = set()
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            if bw > 0 and bh > 0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0 <= cls_id < len(nutrient_classes):
                                valid_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                                img_classes.add(cls_id)
                                split_counts_nutrient["train"]["valid_boxes"] += 1
                            else:
                                split_counts_nutrient["train"]["excluded_boxes"] += 1
                        except ValueError:
                            split_counts_nutrient["train"]["excluded_boxes"] += 1

        shutil.copy2(img_p, yolo_nutrient_dir / "images" / "train" / img_p.name)
        with open(yolo_nutrient_dir / "labels" / "train" / (img_p.stem + ".txt"), "w", encoding="utf-8") as df:
            df.write("\n".join(valid_lines) + "\n" if valid_lines else "")
        split_counts_nutrient["train"]["images"] += 1

        for c in img_classes:
            if len(nutrient_samples_per_class[c]) < 5:
                nutrient_samples_per_class[c].append((img_p, valid_lines))

    # Process val and test
    for img_p in valid_imgs:
        dst_split = "val" if img_p in val_set else "test"
        lbl_p = nutrient_src / "valid" / "labels" / (img_p.stem + ".txt")
        valid_lines = []
        img_classes = set()
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            if bw > 0 and bh > 0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0 <= cls_id < len(nutrient_classes):
                                valid_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                                img_classes.add(cls_id)
                                split_counts_nutrient[dst_split]["valid_boxes"] += 1
                            else:
                                split_counts_nutrient[dst_split]["excluded_boxes"] += 1
                        except ValueError:
                            split_counts_nutrient[dst_split]["excluded_boxes"] += 1

        shutil.copy2(img_p, yolo_nutrient_dir / "images" / dst_split / img_p.name)
        with open(yolo_nutrient_dir / "labels" / dst_split / (img_p.stem + ".txt"), "w", encoding="utf-8") as df:
            df.write("\n".join(valid_lines) + "\n" if valid_lines else "")
        split_counts_nutrient[dst_split]["images"] += 1

        for c in img_classes:
            if len(nutrient_samples_per_class[c]) < 5:
                nutrient_samples_per_class[c].append((img_p, valid_lines))

    # Write data/yolo_nutrient/data.yaml
    nutrient_data_yaml_content = {
        "path": yolo_nutrient_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(nutrient_classes),
        "names": nutrient_classes
    }
    with open(yolo_nutrient_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yaml.dump(nutrient_data_yaml_content, yf, default_flow_style=False)

    # Render Visual Samples for Nutrient
    print("Rendering Visual Validation for Dataset B...")
    for c_id, sample_list in nutrient_samples_per_class.items():
        c_name = nutrient_classes[c_id].replace(" ", "_")
        for idx, (img_p, lines) in enumerate(sample_list):
            im = Image.open(img_p).convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size

            for l in lines:
                cid, xc, yc, bw, bh = l.split()
                cid = int(cid)
                xc, yc, bw, bh = map(float, [xc, yc, bw, bh])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                color = "#33CCFF" if cid == c_id else "#FFCC00"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 3, max(0, y1 - 12)), f"{nutrient_classes[cid]}", fill=color)

            out_fn = f"nutrient_class_{c_id}_{c_name}_sample_{idx+1}.png"
            im.save(nutrient_vis_dir / out_fn)

    # Save summary manifest
    split_summary = {
        "disease_dataset": {
            "splits": dict(split_counts_disease),
            "data_yaml": (yolo_disease_dir / "data.yaml").as_posix()
        },
        "nutrient_dataset": {
            "splits": dict(split_counts_nutrient),
            "data_yaml": (yolo_nutrient_dir / "data.yaml").as_posix()
        }
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("Datasets prepared and visual validation rendered successfully.")

if __name__ == "__main__":
    prepare_and_validate()
