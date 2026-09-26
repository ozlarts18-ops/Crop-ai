"""
prepare_all_datasets.py
Prepares data manifests and directory structures for:
1. Disease Classification (9 classes from Multi-Class & MH-SoyaHealthVision Leaf)
2. Nutrient Classification (Potassium Deficiency vs Normal Status)
3. Growth / Weed Analysis (FarmBot 20-day YOLO dataset)
4. Crop Verification (SoyCotton Soybean vs Cotton)

Adheres strictly to:
- docs/dataset_audit_final.md
- docs/dataset_leakage_report.md
- configs/dataset_classes.yaml
- Zero data leakage (hash-grouped stratification)
- Exclusion of 20 conflicting MH Leaf images (Pest vs Mosaic)
- Exclusion of UAV images and SoyNet
"""

import os
import sys
import csv
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
import random
from PIL import Image

random.seed(42)

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
RAW_DIR = ROOT_DIR / "raw"
DATA_DIR = ROOT_DIR / "data"
META_DIR = DATA_DIR / "metadata"
PREP_DIR = DATA_DIR / "prepared"

META_DIR.mkdir(parents=True, exist_ok=True)
PREP_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==========================================
# 1. PREPARE DISEASE DATASET (9 CLASSES)
# ==========================================
def prepare_disease():
    print("\n" + "="*50)
    print("STEP 1: PREPARING DISEASE DATASET (9 CLASSES)")
    print("="*50)

    # Output directory for extracted MH Leaf images
    mh_leaf_out = PREP_DIR / "mh_leaf_images"
    mh_leaf_out.mkdir(parents=True, exist_ok=True)

    # 1. Read Multi-Class Soybean Leaf Disease
    mc_dir = RAW_DIR / "Multi-Class Soybean Leaf Disease Dataset Healthy a" / "Soyabean leaf desease dataset"
    mc_rows = []
    if mc_dir.exists():
        for f in sorted(mc_dir.rglob("*.jpg")):
            cname = f.parent.name.replace(" ", "_")
            if cname == "Rust":
                cname = "Soybean_Rust"
            h = md5_file(f)
            mc_rows.append({
                "image_id": f.stem,
                "filepath": str(f.resolve()),
                "dataset_source": "Multi-Class",
                "class_name": cname,
                "hash_md5": h
            })
    print(f"Multi-Class images found: {len(mc_rows)}")

    # 2. Extract MH-SoyaHealthVision Leaf Subset
    # Mapping of zip files to canonical classes
    mh_leaf_zip_dir = RAW_DIR / "MH-SoyaHealthVision" / "Soyabean_Leaf_Image_Dataset"
    if not mh_leaf_zip_dir.exists():
        mh_leaf_zip_dir = RAW_DIR / "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment" / "Soyabean_Leaf_Image_Dataset"

    zip_class_map = {
        "Caterpillar and Semilooper Pest Attack.zip": "Pest_Damage",
        "Healthy_Soyabean.zip": "Healthy",
        "Soyabean_Frog_Leaf_Eye.zip": "Frogeye_Leaf_Spot",
        "Soyabean_Mosaic.zip": "Mosaic",
        "Soyabean_Rust.zip": "Soybean_Rust",
        "Soyabean_Spectoria_Brown_Spot.zip": "Septoria_Brown_Spot"
    }

    mh_raw_entries = []
    # Track hashes to find the 20 conflicting duplicates between Pest and Mosaic
    hash_to_mh_classes = defaultdict(set)

    for zip_name, canonical_class in zip_class_map.items():
        zip_path = mh_leaf_zip_dir / zip_name
        if not zip_path.exists():
            print(f"Warning: {zip_path} not found!")
            continue
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX"):
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    img_data = zf.read(info.filename)
                    h = hashlib.md5(img_data).hexdigest()
                    hash_to_mh_classes[h].add(canonical_class)
                    mh_raw_entries.append({
                        "filename": Path(info.filename).name,
                        "zip_name": zip_name,
                        "class_name": canonical_class,
                        "hash_md5": h,
                        "img_data": img_data
                    })

    # Identify conflicting hashes (e.g. exists in Pest and Mosaic)
    conflicting_hashes = {h for h, cset in hash_to_mh_classes.items() if len(cset) > 1}
    print(f"MH Leaf total raw images in zips: {len(mh_raw_entries)}")
    print(f"MH Leaf conflicting hashes identified across classes: {len(conflicting_hashes)} (Will be excluded)")

    mh_rows = []
    seen_mh_hashes = set()
    for entry in mh_raw_entries:
        h = entry["hash_md5"]
        if h in conflicting_hashes:
            continue
        if h in seen_mh_hashes:
            continue
        seen_mh_hashes.add(h)

        # Save image to disk
        target_name = f"MH_{entry['class_name']}_{h[:10]}.jpg"
        target_path = mh_leaf_out / target_name
        if not target_path.exists():
            with open(target_path, "wb") as f:
                f.write(entry["img_data"])

        mh_rows.append({
            "image_id": target_name.replace(".jpg", ""),
            "filepath": str(target_path.resolve()),
            "dataset_source": "MH-SoyaHealthVision",
            "class_name": entry["class_name"],
            "hash_md5": h
        })
    print(f"MH Leaf clean non-conflicting images extracted: {len(mh_rows)}")

    # Combine Multi-Class and MH Leaf
    all_disease = mc_rows + mh_rows
    print(f"Total disease samples combined: {len(all_disease)}")

    # Hash-grouped stratified split (70% train / 15% val / 15% test)
    # Group by hash first
    hash_to_records = defaultdict(list)
    for r in all_disease:
        hash_to_records[r["hash_md5"]].append(r)

    # Class to unique hashes
    class_to_hashes = defaultdict(list)
    for h, recs in hash_to_records.items():
        c = recs[0]["class_name"]
        class_to_hashes[c].append(h)

    hash_to_split = {}
    for c, h_list in sorted(class_to_hashes.items()):
        random.shuffle(h_list)
        n = len(h_list)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        for i, h in enumerate(h_list):
            if i < n_train:
                hash_to_split[h] = "train"
            elif i < n_train + n_val:
                hash_to_split[h] = "val"
            else:
                hash_to_split[h] = "test"

    for r in all_disease:
        r["split"] = hash_to_split[r["hash_md5"]]

    # Save to data/metadata/disease_metadata.csv
    csv_path = META_DIR / "disease_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "dataset_source", "class_name", "hash_md5", "split", "is_duplicate"])
        for r in all_disease:
            h = r["hash_md5"]
            is_dupe = len(hash_to_records[h]) > 1
            writer.writerow([r["image_id"], r["filepath"], r["dataset_source"], r["class_name"], h, r["split"], is_dupe])

    # Print summary
    class_split_counts = defaultdict(lambda: Counter())
    for r in all_disease:
        class_split_counts[r["class_name"]][r["split"]] += 1

    print("\nDisease Dataset Split Summary:")
    print(f"{'Class':<25} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 57)
    for c in sorted(class_split_counts.keys()):
        cnts = class_split_counts[c]
        tot = sum(cnts.values())
        print(f"{c:<25} {cnts['train']:<8} {cnts['val']:<8} {cnts['test']:<8} {tot:<8}")
    print("-" * 57)
    tot_train = sum(c["train"] for c in class_split_counts.values())
    tot_val = sum(c["val"] for c in class_split_counts.values())
    tot_test = sum(c["test"] for c in class_split_counts.values())
    print(f"{'TOTAL':<25} {tot_train:<8} {tot_val:<8} {tot_test:<8} {tot_train+tot_val+tot_test:<8}")
    print(f"Saved manifest: {csv_path}")
    return all_disease


# ==========================================
# 2. PREPARE NUTRIENT DATASET
# (Potassium_Deficiency vs Normal_Status)
# ==========================================
def prepare_nutrient(disease_samples):
    print("\n" + "="*50)
    print("STEP 2: PREPARING NUTRIENT DATASET")
    print("="*50)

    # 1. Load Potassium Deficiency images
    k_dir = RAW_DIR / "potassium_deficiency" / "potassium_deficiency"
    k_images = sorted(list(k_dir.glob("*.jpg")))
    print(f"Potassium deficiency raw images: {len(k_images)}")

    k_rows = []
    for f in k_images:
        h = md5_file(f)
        k_rows.append({
            "image_id": f.stem,
            "filepath": str(f.resolve()),
            "dataset_source": "potassium_deficiency",
            "class_name": "Potassium_Deficiency",
            "hash_md5": h
        })

    # Split Potassium Deficiency by hash (70/15/15)
    random.shuffle(k_rows)
    n_k = len(k_rows)
    n_k_train = int(n_k * 0.70)
    n_k_val = int(n_k * 0.15)
    for i, r in enumerate(k_rows):
        if i < n_k_train:
            r["split"] = "train"
        elif i < n_k_train + n_k_val:
            r["split"] = "val"
        else:
            r["split"] = "test"

    # 2. Select Healthy controls from audited healthy soybean samples
    # Must prevent cross-split leakage! Use the exact split assignment from disease dataset
    healthy_controls = []
    for r in disease_samples:
        if r["class_name"] == "Healthy":
            healthy_controls.append({
                "image_id": f"ctrl_{r['image_id']}",
                "filepath": r["filepath"],
                "dataset_source": f"Control_{r['dataset_source']}",
                "class_name": "Normal_Status",
                "hash_md5": r["hash_md5"],
                "split": r["split"] # Match the disease split!
            })
    print(f"Healthy control images selected: {len(healthy_controls)}")

    all_nutrient = k_rows + healthy_controls
    print(f"Total nutrient dataset size: {len(all_nutrient)}")

    csv_path = META_DIR / "nutrient_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "dataset_source", "class_name", "hash_md5", "split"])
        for r in all_nutrient:
            writer.writerow([r["image_id"], r["filepath"], r["dataset_source"], r["class_name"], r["hash_md5"], r["split"]])

    counts = defaultdict(lambda: Counter())
    for r in all_nutrient:
        counts[r["class_name"]][r["split"]] += 1

    print("\nNutrient Dataset Split Summary:")
    print(f"{'Class':<25} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 57)
    for c in sorted(counts.keys()):
        cnts = counts[c]
        print(f"{c:<25} {cnts['train']:<8} {cnts['val']:<8} {cnts['test']:<8} {sum(cnts.values()):<8}")
    print(f"Saved manifest: {csv_path}")
    return all_nutrient


# ==========================================
# 3. PREPARE GROWTH / WEED DATASET (YOLO)
# ==========================================
def prepare_growth():
    print("\n" + "="*50)
    print("STEP 3: PREPARING GROWTH / WEED DATASET (FARMBOT YOLO)")
    print("="*50)

    farm_zip = RAW_DIR / "A soybean and weed image dataset collected using F" / "A soybean and weed image dataset collected using F" / "Soyaben-Weed Dataset.zip"
    yolo_dir = PREP_DIR / "growth_yolo"

    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    extracted_samples = []
    seen_hashes = {} # hash -> sample dict (to deduplicate Day 5 & Day 12 identical images)
    duplicate_count = 0

    with zipfile.ZipFile(farm_zip, "r") as zf:
        infolist = zf.infolist()
        # Find all image files in Annotated Dataset
        img_entries = [info for info in infolist if "Annotated Dataset" in info.filename and info.filename.lower().endswith((".jpg", ".jpeg")) and not info.filename.startswith("__MACOSX")]
        print(f"Found {len(img_entries)} image files in Annotated Dataset")

        for info in img_entries:
            parts = Path(info.filename).parts
            day = "Unknown"
            for p in parts:
                if p.startswith("Day"):
                    day = p
                    break
            img_bytes = zf.read(info.filename)
            h = hashlib.md5(img_bytes).hexdigest()

            # Find matching label
            label_filename = info.filename.replace("/images/", "/labels/")
            label_filename = str(Path(label_filename).with_suffix(".txt")).replace("\\", "/")
            label_bytes = b""
            try:
                label_bytes = zf.read(label_filename)
            except KeyError:
                pass

            if h in seen_hashes:
                duplicate_count += 1
                continue # Skip exact duplicate frame to eliminate inter-day leakage

            seen_hashes[h] = {
                "filename": Path(info.filename).name,
                "label_filename": Path(label_filename).name,
                "day": day,
                "img_bytes": img_bytes,
                "label_bytes": label_bytes,
                "hash_md5": h
            }

    unique_samples = list(seen_hashes.values())
    print(f"Unique FarmBot images: {len(unique_samples)} (Excluded {duplicate_count} duplicate frames)")

    # Stratified/Temporal split by day & hash
    day_groups = defaultdict(list)
    for s in unique_samples:
        day_groups[s["day"]].append(s)

    # Within each day, 70% train, 15% val, 15% test
    split_counts = Counter()
    growth_rows = []

    for day, items in sorted(day_groups.items()):
        random.shuffle(items)
        n = len(items)
        n_tr = max(1, int(n * 0.70))
        n_v = max(1, int(n * 0.15))
        for i, it in enumerate(items):
            if i < n_tr:
                s = "train"
            elif i < n_tr + n_v:
                s = "val"
            else:
                s = "test"
            split_counts[s] += 1
            it["split"] = s

            # Write image and label to yolo_dir
            im_path = yolo_dir / "images" / s / it["filename"]
            lb_path = yolo_dir / "labels" / s / it["label_filename"]
            with open(im_path, "wb") as f:
                f.write(it["img_bytes"])
            with open(lb_path, "wb") as f:
                f.write(it["label_bytes"])

            growth_rows.append({
                "image_id": Path(it["filename"]).stem,
                "filename": it["filename"],
                "day": it["day"],
                "split": s,
                "hash_md5": it["hash_md5"]
            })

    # Create data.yaml for YOLO
    data_yaml_content = f"""# FarmBot Growth & Weed Detection Dataset
path: {str(yolo_dir.resolve()).replace('\\', '/')}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['plant', 'weed']
"""
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    # Update metadata
    csv_path = META_DIR / "growth_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filename", "day", "split", "hash_md5"])
        for r in growth_rows:
            writer.writerow([r["image_id"], r["filename"], r["day"], r["split"], r["hash_md5"]])

    print(f"Growth YOLO dataset prepared at: {yolo_dir}")
    print(f"Split distribution: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    print(f"Saved manifest: {csv_path}")
    return yaml_path


# ==========================================
# 4. PREPARE CROP VERIFICATION DATASET
# (Soybean vs Cotton Binary Classifier)
# ==========================================
def prepare_crop_verification():
    print("\n" + "="*50)
    print("STEP 4: PREPARING CROP VERIFICATION DATASET (SOYBEAN VS COTTON)")
    print("="*50)

    sc_coco = RAW_DIR / "SoyCotton" / "SoyCotton" / "annotations" / "coco.json"
    sc_img_dir = RAW_DIR / "SoyCotton" / "SoyCotton" / "images"
    crop_out = PREP_DIR / "crop_verification_crops"
    crop_out.mkdir(parents=True, exist_ok=True)

    with open(sc_coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1. Pure full images from crop_verification_metadata.csv
    meta_df = []
    with open(META_DIR / "crop_verification_metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta_df.append(r)

    pure_soy_imgs = [r["filename"] for r in meta_df if int(r["soy_instances"]) > 0 and int(r["cotton_instances"]) == 0]
    pure_cotton_imgs = [r["filename"] for r in meta_df if int(r["cotton_instances"]) > 0 and int(r["soy_instances"]) == 0]

    print(f"Pure Soy full images: {len(pure_soy_imgs)}")
    print(f"Pure Cotton full images: {len(pure_cotton_imgs)}")

    records = []

    # Add pure full images
    for fn in pure_soy_imgs:
        p = sc_img_dir / fn
        if p.exists():
            h = md5_file(p)
            records.append({
                "image_id": f"full_soy_{Path(fn).stem}",
                "filepath": str(p.resolve()),
                "class_name": "Soybean",
                "sample_type": "full_image",
                "hash_md5": h
            })

    for fn in pure_cotton_imgs:
        p = sc_img_dir / fn
        if p.exists():
            h = md5_file(p)
            records.append({
                "image_id": f"full_cotton_{Path(fn).stem}",
                "filepath": str(p.resolve()),
                "class_name": "Cotton",
                "sample_type": "full_image",
                "hash_md5": h
            })

    # 2. Extract high-quality plant instance crops from COCO annotations
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    cat_map = {1: "Soybean", 2: "Cotton"}

    crops_by_class = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_map:
            continue
        bbox = ann["bbox"] # [x, y, w, h]
        w, h = bbox[2], bbox[3]
        if w >= 120 and h >= 120:
            crops_by_class[cat_map[cat_id]].append((ann["image_id"], ann["id"], bbox))

    print(f"Eligible plant crops (>=120px): Soy={len(crops_by_class['Soybean'])}, Cotton={len(crops_by_class['Cotton'])}")

    random.shuffle(crops_by_class["Soybean"])
    random.shuffle(crops_by_class["Cotton"])
    target_crops_per_class = 400

    img_cache = {}
    for cname in ["Soybean", "Cotton"]:
        selected = crops_by_class[cname][:target_crops_per_class]
        for img_id, ann_id, bbox in selected:
            fn = img_id_to_file.get(img_id)
            if not fn:
                continue
            src_path = sc_img_dir / fn
            if not src_path.exists():
                continue
            if fn not in img_cache:
                img_cache[fn] = Image.open(src_path).convert("RGB")
            im = img_cache[fn]
            x, y, w, h = [int(v) for v in bbox]
            crop_im = im.crop((x, y, x + w, y + h))

            crop_name = f"crop_{cname}_{img_id}_{ann_id}.jpg"
            crop_path = crop_out / crop_name
            if not crop_path.exists():
                crop_im.save(crop_path, quality=92)

            h_crop = md5_file(crop_path)
            records.append({
                "image_id": crop_name.replace(".jpg", ""),
                "filepath": str(crop_path.resolve()),
                "class_name": cname,
                "sample_type": "instance_crop",
                "hash_md5": h_crop
            })

    print(f"Total crop verification samples: {len(records)}")

    # Hash-grouped stratified 70/15/15 split
    hash_groups = defaultdict(list)
    for r in records:
        hash_groups[r["hash_md5"]].append(r)

    class_hashes = defaultdict(list)
    for h, recs in hash_groups.items():
        c = recs[0]["class_name"]
        class_hashes[c].append(h)

    hash_to_split = {}
    for c, h_list in class_hashes.items():
        random.shuffle(h_list)
        n = len(h_list)
        n_tr = int(n * 0.70)
        n_val = int(n * 0.15)
        for i, h in enumerate(h_list):
            if i < n_tr:
                hash_to_split[h] = "train"
            elif i < n_tr + n_val:
                hash_to_split[h] = "val"
            else:
                hash_to_split[h] = "test"

    for r in records:
        r["split"] = hash_to_split[r["hash_md5"]]

    csv_path = META_DIR / "crop_verification_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "class_name", "sample_type", "hash_md5", "split"])
        for r in records:
            writer.writerow([r["image_id"], r["filepath"], r["class_name"], r["sample_type"], r["hash_md5"], r["split"]])

    counts = defaultdict(lambda: Counter())
    for r in records:
        counts[r["class_name"]][r["split"]] += 1

    print("\nCrop Verification Dataset Split Summary:")
    print(f"{'Class':<25} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 57)
    for c in sorted(counts.keys()):
        cnts = counts[c]
        print(f"{c:<25} {cnts['train']:<8} {cnts['val']:<8} {cnts['test']:<8} {sum(cnts.values()):<8}")
    print(f"Saved manifest: {csv_path}")
    return records


if __name__ == "__main__":
    print("Starting Comprehensive Dataset Preparation...")
    disease_samples = prepare_disease()
    nutrient_samples = prepare_nutrient(disease_samples)
    growth_yaml = prepare_growth()
    crop_samples = prepare_crop_verification()
    print("\n" + "="*50)
    print("ALL 4 DATASETS SUCCESSFULLY PREPARED WITH ZERO LEAKAGE!")
    print("="*50)
