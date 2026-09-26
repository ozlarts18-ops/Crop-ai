"""
Generate Metadata Manifests for Crop_AI Prepared Tasks
Creates:
- data/metadata/disease_metadata.csv
- data/metadata/nutrient_metadata.csv
- data/metadata/growth_metadata.csv
- data/metadata/crop_verification_metadata.csv
"""

import os
import csv
import json
import hashlib
import zipfile
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
RAW_DIR = ROOT_DIR / "raw"
DATA_DIR = ROOT_DIR / "data"
META_DIR = DATA_DIR / "metadata"
PREP_DIR = DATA_DIR / "prepared"

META_DIR.mkdir(parents=True, exist_ok=True)
for task in ["disease", "nutrient", "growth", "crop_verification"]:
    for split in ["train", "val", "test"]:
        (PREP_DIR / task / split).mkdir(parents=True, exist_ok=True)

# 1. Disease Metadata (Multi-Class + MH-SoyaHealthVision Leaf)
print("Building disease_metadata.csv...")
disease_rows = []

# Multi-Class
mc_dir = RAW_DIR / "Multi-Class Soybean Leaf Disease Dataset Healthy a" / "Soyabean leaf desease dataset"
if mc_dir.exists():
    for f in sorted(mc_dir.rglob('*.jpg')):
        rel = f.relative_to(ROOT_DIR)
        cname = f.parent.name
        # standardize
        cname_clean = cname.replace(' ', '_')
        if cname_clean == 'Rust': cname_clean = 'Soybean_Rust'
        h = hashlib.md5(f.read_bytes()).hexdigest()
        disease_rows.append({
            "image_id": f.stem,
            "relative_path": str(rel).replace('\\', '/'),
            "dataset_source": "Multi-Class",
            "class_name": cname_clean,
            "hash_md5": h
        })

# Deduplicate & Split Disease
hash_groups = defaultdict(list)
for r in disease_rows:
    hash_groups[r["hash_md5"]].append(r)

# Stratified split by hash group
class_to_hashes = defaultdict(set)
for r in disease_rows:
    class_to_hashes[r["class_name"]].add(r["hash_md5"])

hash_to_split = {}
for cname, h_set in class_to_hashes.items():
    h_list = sorted(list(h_set))
    random.shuffle(h_list)
    n = len(h_list)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    for i, h in enumerate(h_list):
        if i < n_train:
            hash_to_split[h] = "train"
        elif i < n_train + n_val:
            hash_to_split[h] = "val"
        else:
            hash_to_split[h] = "test"

with open(META_DIR / "disease_metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "relative_path", "dataset_source", "class_name", "hash_md5", "split", "is_duplicate"])
    for r in disease_rows:
        h = r["hash_md5"]
        is_dupe = len(hash_groups[h]) > 1
        split = hash_to_split.get(h, "train")
        writer.writerow([r["image_id"], r["relative_path"], r["dataset_source"], r["class_name"], h, split, is_dupe])

# 2. Nutrient Metadata (potassium_deficiency)
print("Building nutrient_metadata.csv...")
k_rows = []
kp = RAW_DIR / "potassium_deficiency" / "potassium_deficiency"
if kp.exists():
    for f in sorted(kp.glob('*.jpg')):
        rel = f.relative_to(ROOT_DIR)
        h = hashlib.md5(f.read_bytes()).hexdigest()
        k_rows.append({
            "image_id": f.stem,
            "relative_path": str(rel).replace('\\', '/'),
            "dataset_source": "potassium_deficiency",
            "class_name": "Potassium_Deficiency",
            "hash_md5": h
        })

random.shuffle(k_rows)
n = len(k_rows)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
for i, r in enumerate(k_rows):
    if i < n_train:
        r["split"] = "train"
    elif i < n_train + n_val:
        r["split"] = "val"
    else:
        r["split"] = "test"

with open(META_DIR / "nutrient_metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "relative_path", "dataset_source", "class_name", "hash_md5", "split"])
    for r in k_rows:
        writer.writerow([r["image_id"], r["relative_path"], r["dataset_source"], r["class_name"], r["hash_md5"], r["split"]])

# 3. Growth Metadata (FarmBot Dataset)
print("Building growth_metadata.csv...")
farm_zip = RAW_DIR / "A soybean and weed image dataset collected using F" / "A soybean and weed image dataset collected using F" / "Soyaben-Weed Dataset.zip"
growth_rows = []
if farm_zip.exists():
    with zipfile.ZipFile(farm_zip, 'r') as zf:
        for info in zf.infolist():
            if info.filename.startswith('__MACOSX') or info.is_dir():
                continue
            fn = info.filename
            ext = Path(fn).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                h = hashlib.md5(zf.read(fn)).hexdigest()
                is_annotated = "Annotated Dataset" in fn
                day = "Unspecified"
                if is_annotated:
                    for part in Path(fn).parts:
                        if "Day" in part:
                            day = part
                            break
                growth_rows.append({
                    "image_id": Path(fn).stem,
                    "archive_path": fn,
                    "day": day,
                    "is_annotated": is_annotated,
                    "hash_md5": h
                })

with open(META_DIR / "growth_metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "archive_path", "day", "is_annotated", "hash_md5"])
    for r in growth_rows:
        writer.writerow([r["image_id"], r["archive_path"], r["day"], r["is_annotated"], r["hash_md5"]])

# 4. Crop Verification Metadata (SoyCotton)
print("Building crop_verification_metadata.csv...")
sc_coco = RAW_DIR / "SoyCotton" / "SoyCotton" / "annotations" / "coco.json"
sc_rows = []
if sc_coco.exists():
    with open(sc_coco, 'r') as f:
        coco = json.load(f)
    img_id_to_counts = defaultdict(lambda: {"soy": 0, "cotton": 0})
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id == 1:
            img_id_to_counts[iid]["soy"] += 1
        elif cat_id == 2:
            img_id_to_counts[iid]["cotton"] += 1
            
    for img_info in coco.get("images", []):
        iid = img_info["id"]
        fn = img_info["file_name"]
        sc_rows.append({
            "image_id": iid,
            "filename": fn,
            "relative_path": f"raw/SoyCotton/SoyCotton/images/{fn}",
            "soy_instances": img_id_to_counts[iid]["soy"],
            "cotton_instances": img_id_to_counts[iid]["cotton"]
        })

with open(META_DIR / "crop_verification_metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "filename", "relative_path", "soy_instances", "cotton_instances"])
    for r in sc_rows:
        writer.writerow([r["image_id"], r["filename"], r["relative_path"], r["soy_instances"], r["cotton_instances"]])

print("All metadata manifests generated successfully in data/metadata/!")
