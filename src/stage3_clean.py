import os
import io
import csv
import json
import zipfile
import shutil
from PIL import Image, UnidentifiedImageError
from src.config import (
    RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, NORMALIZED_CLASSES, CLASS_NAME_TO_ID
)
from src.utils import compute_sha256, compute_dhash, hamming_distance

RAW_CLASS_MAPPING = {
    # Multi-Class Soybean Leaf Disease Dataset
    "Bacterial Blight": "Bacterial_Blight",
    "Cercospora Leaf Blight": "Cercospora_Leaf_Blight",
    "Healthy": "Healthy",
    "Rust": "Soybean_Rust",
    "Sudden Death Syndrome": "Sudden_Death_Syndrome",
    
    # MH-SoyaHealthVision
    "Caterpillar and Semilooper Pest Attack": "Pest_Damage",
    "Healthy_Soyabean": "Healthy",
    "Soyabean_Frog_Leaf_Eye": "Frogeye_Leaf_Spot",
    "Soyabean_Mosaic": "Mosaic",
    "Soyabean_Rust": "Soybean_Rust",
    "Soyabean_Spectoria_Brown_Spot": "Septoria_Brown_Spot",
    "Soyabean Semilooper and Caterpillar_Pest_Attack": "Pest_Damage",
    "rust": "Soybean_Rust",
    
    # SoyNet
    "Healthy_pic": "Healthy",
    "Grayscale_Healthy_data": "Healthy",
    "Disease_Pic": "UNKNOWN",
    "Grayscale_Disease_data": "UNKNOWN",
    "Disease_Preprocessing data": "UNKNOWN",
    "Mobile pic": "UNKNOWN",
    "Mobile Click_256_256": "UNKNOWN",
    "GrayScale_Mobile Click": "UNKNOWN"
}

def run_stage3(max_images_per_class=None):
    print("Executing Stage 3: Clean & Prepare Data (Quality Audit, SHA256 & Perceptual Hashing)...")
    
    clean_images_dir = os.path.join(PROCESSED_DIR, "clean_images")
    os.makedirs(clean_images_dir, exist_ok=True)
    
    corrupted_images = []
    invalid_annotations = []
    unknown_classes = []
    duplicates = []
    
    seen_sha256 = {}  # sha256 -> first_seen_record
    seen_dhash = {}   # dhash -> first_seen_record
    
    valid_clean_records = []
    
    total_scanned = 0
    
    # Helper to process an image from path or bytes
    def inspect_and_register(img_bytes, filename, source_dataset, raw_class, rel_path, image_type="leaf"):
        nonlocal total_scanned
        total_scanned += 1
        
        # Check corruption
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.verify()
            # Reopen for actual dimension/hash
            pil_img = Image.open(io.BytesIO(img_bytes))
            width, height = pil_img.size
            img_format = pil_img.format
            mode = pil_img.mode
        except Exception as e:
            corrupted_images.append({
                "filename": filename,
                "source_dataset": source_dataset,
                "path": rel_path,
                "error": str(e)
            })
            return
            
        sha256_val = compute_sha256(img_bytes)
        dhash_val = compute_dhash(pil_img)
        
        # Check duplicate
        if sha256_val in seen_sha256:
            duplicates.append({
                "type": "exact_sha256",
                "filename": filename,
                "original_filename": seen_sha256[sha256_val]["filename"],
                "source_dataset": source_dataset,
                "original_dataset": seen_sha256[sha256_val]["source_dataset"],
                "sha256": sha256_val,
                "dhash": dhash_val
            })
            return
            
        # Check near duplicate (dhash match)
        if dhash_val in seen_dhash:
            duplicates.append({
                "type": "near_duplicate_dhash",
                "filename": filename,
                "original_filename": seen_dhash[dhash_val]["filename"],
                "source_dataset": source_dataset,
                "original_dataset": seen_dhash[dhash_val]["source_dataset"],
                "sha256": sha256_val,
                "dhash": dhash_val
            })
            return
            
        # Class mapping
        norm_class = RAW_CLASS_MAPPING.get(raw_class, "UNKNOWN")
        if norm_class == "UNKNOWN":
            unknown_classes.append({
                "filename": filename,
                "source_dataset": source_dataset,
                "raw_class": raw_class,
                "path": rel_path
            })
            return
            
        # Register seen hashes
        rec = {
            "filename": filename,
            "source_dataset": source_dataset,
            "raw_class": raw_class,
            "norm_class": norm_class,
            "norm_class_id": CLASS_NAME_TO_ID[norm_class],
            "sha256": sha256_val,
            "dhash": dhash_val,
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "image_type": image_type
        }
        seen_sha256[sha256_val] = rec
        seen_dhash[dhash_val] = rec
        
        # Save clean processed image copy
        clean_name = f"{source_dataset[:3].upper()}_{rec['norm_class_id']}_{sha256_val[:10]}.jpg"
        dest_path = os.path.join(clean_images_dir, clean_name)
        
        # Convert to RGB and save standard JPEG
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.save(dest_path, "JPEG", quality=95)
        
        rec["processed_path"] = dest_path
        rec["processed_filename"] = clean_name
        valid_clean_records.append(rec)

    # 1. Process Multi-Class Soybean Leaf Disease Dataset
    print("  Auditing Multi-Class Soybean Leaf Disease Dataset...")
    mc_dir = os.path.join(RAW_DIR, "Multi-Class Soybean Leaf Disease Dataset Healthy a")
    for root, dirs, files in os.walk(mc_dir):
        raw_class = os.path.basename(root)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fp = os.path.join(root, f)
                with open(fp, "rb") as bf:
                    inspect_and_register(
                        bf.read(), f, "Multi-Class-Soybean", raw_class,
                        os.path.relpath(fp, RAW_DIR), image_type="leaf"
                    )

    # 2. Process MH-SoyaHealthVision
    print("  Auditing MH-SoyaHealthVision archives...")
    mh_dir = os.path.join(RAW_DIR, "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment")
    for root, dirs, files in os.walk(mh_dir):
        is_uav = "UAV" in root
        img_type = "uav" if is_uav else "leaf"
        for f in files:
            if f.endswith('.zip'):
                zip_path = os.path.join(root, f)
                zip_class = os.path.splitext(f)[0]
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            base_n = os.path.basename(name)
                            if base_n:
                                img_bytes = zf.read(name)
                                inspect_and_register(
                                    img_bytes, base_n, "MH-SoyaHealthVision",
                                    zip_class, name, image_type=img_type
                                )

    # 3. Process SoyNet
    print("  Auditing SoyNet dataset...")
    sn_dir = os.path.join(RAW_DIR, "SoyNet Indian Soybean Image dataset with quality i")
    for root, dirs, files in os.walk(sn_dir):
        folder_name = os.path.basename(root)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fp = os.path.join(root, f)
                with open(fp, "rb") as bf:
                    inspect_and_register(
                        bf.read(), f, "SoyNet", folder_name,
                        os.path.relpath(fp, RAW_DIR), image_type="leaf"
                    )

    # Record any invalid annotations
    # (Since 0 bounding boxes or segmentation masks exist in source, any expected localization annotation is missing)
    invalid_annotations.append({
        "dataset": "All",
        "description": "Source datasets do not supply bounding boxes or segmentation masks; localization annotations unavailable."
    })

    print(f"Audit summary: Total Scanned: {total_scanned}, Clean Unique: {len(valid_clean_records)}, Duplicates: {len(duplicates)}, Corrupted: {len(corrupted_images)}, Unknown Classes: {len(unknown_classes)}")

    # Write output CSVs and JSONs
    # 1. outputs/corrupted_images.csv
    with open(os.path.join(OUTPUTS_DIR, "corrupted_images.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_dataset", "path", "error"])
        writer.writeheader()
        writer.writerows(corrupted_images)

    # 2. outputs/duplicates.csv
    with open(os.path.join(OUTPUTS_DIR, "duplicates.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "filename", "original_filename", "source_dataset", "original_dataset", "sha256", "dhash"])
        writer.writeheader()
        writer.writerows(duplicates)

    # 3. outputs/unknown_classes.csv
    with open(os.path.join(OUTPUTS_DIR, "unknown_classes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_dataset", "raw_class", "path"])
        writer.writeheader()
        writer.writerows(unknown_classes)

    # 4. outputs/invalid_annotations.csv
    with open(os.path.join(OUTPUTS_DIR, "invalid_annotations.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "description"])
        writer.writeheader()
        writer.writerows(invalid_annotations)

    # 5. outputs/data_quality_report.json
    quality_report = {
        "total_images_scanned": total_scanned,
        "clean_unique_images": len(valid_clean_records),
        "corrupted_images_count": len(corrupted_images),
        "duplicate_images_count": len(duplicates),
        "unknown_classes_count": len(unknown_classes),
        "class_distribution_clean": {}
    }
    for r in valid_clean_records:
        cls_name = r["norm_class"]
        quality_report["class_distribution_clean"][cls_name] = quality_report["class_distribution_clean"].get(cls_name, 0) + 1

    with open(os.path.join(OUTPUTS_DIR, "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=4)

    # 6. outputs/data_quality_report.csv
    with open(os.path.join(OUTPUTS_DIR, "data_quality_report.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in quality_report.items():
            if k != "class_distribution_clean":
                writer.writerow([k, v])
        for k, v in quality_report["class_distribution_clean"].items():
            writer.writerow([f"class_{k}", v])

    # Save valid clean records intermediate json for stage 4
    with open(os.path.join(PROCESSED_DIR, "clean_records.json"), "w", encoding="utf-8") as f:
        json.dump(valid_clean_records, f, indent=2)

    print(f"Stage 3 complete: Created data/processed/clean_images and all quality reports.")
    return valid_clean_records

if __name__ == "__main__":
    run_stage3()
