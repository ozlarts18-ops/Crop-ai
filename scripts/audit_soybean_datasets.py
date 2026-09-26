"""
Comprehensive Soybean Dataset Audit Script
Audits remaining datasets:
1. MH-SoyaHealthVision
2. Multi-Class Soybean Leaf Disease Dataset Healthy a
3. SoyCotton
4. SoyNet Indian Soybean Image dataset with quality i
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image
import numpy as np

RAW_DIR = Path("c:/Users/oswal/Music/Crop_AI/raw")

def get_image_info(img_path):
    try:
        with Image.open(img_path) as img:
            return img.size, img.format # (width, height), format
    except Exception:
        return None, None

def audit_dataset_structure(dataset_path: Path):
    print(f"\n==========================================")
    print(f"AUDITING: {dataset_path.name}")
    print(f"PATH: {dataset_path.resolve()}")
    print(f"==========================================")
    
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    all_files = list(dataset_path.rglob('*'))
    file_count = len([f for f in all_files if f.is_file()])
    
    img_files = [f for f in all_files if f.is_file() and f.suffix.lower() in img_exts]
    non_img_files = [f for f in all_files if f.is_file() and f.suffix.lower() not in img_exts]
    
    print(f"Total files: {file_count}")
    print(f"Image files: {len(img_files)}")
    print(f"Non-image files: {len(non_img_files)}")
    
    # Extensions breakdown
    ext_counts = defaultdict(int)
    for f in all_files:
        if f.is_file():
            ext_counts[f.suffix.lower()] += 1
    print("File extensions:", dict(ext_counts))
    
    # Non-image files breakdown
    annotation_files = []
    for f in non_img_files:
        if f.suffix.lower() in {'.json', '.xml', '.txt', '.csv', '.yaml', '.yml'}:
            annotation_files.append(f)
    print(f"Annotation-like files found: {len(annotation_files)}")
    if annotation_files[:10]:
        print("Sample annotation files:", [str(f.relative_to(dataset_path)) for f in annotation_files[:10]])

    # Directory hierarchy / classes
    top_subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Top-level subdirectories: {[d.name for d in top_subdirs]}")
    
    # Class mapping by folder structure
    # Check if subfolders represent classes
    class_images = defaultdict(list)
    for img in img_files:
        rel = img.relative_to(dataset_path)
        parts = rel.parts
        if len(parts) > 1:
            class_key = parts[0]
            # If parts[0] is train/test/val, take parts[1]
            if class_key.lower() in {'train', 'test', 'val', 'valid', 'validation', 'dataset', 'images'}:
                if len(parts) > 2:
                    class_key = f"{parts[0]}/{parts[1]}"
            class_images[class_key].append(img)
        else:
            class_images['<root>'].append(img)
            
    print(f"Identified structure keys / classes ({len(class_images)}):")
    for k, imgs in sorted(class_images.items()):
        print(f"  - {k}: {len(imgs)} images")
        
    # Sample dimensions
    sample_imgs = img_files[:50]
    dims = []
    for s in sample_imgs:
        dim, fmt = get_image_info(s)
        if dim:
            dims.append(dim)
    if dims:
        ws = [d[0] for d in dims]
        hs = [d[1] for d in dims]
        print(f"Sample image dimensions: min=({min(ws)}x{min(hs)}), max=({max(ws)}x{max(hs)}), sample_shape={dims[0]}")
        
    return {
        "name": dataset_path.name,
        "path": str(dataset_path.resolve()),
        "total_files": file_count,
        "image_count": len(img_files),
        "ext_counts": dict(ext_counts),
        "classes": {k: len(v) for k, v in class_images.items()},
        "img_paths": img_files
    }

if __name__ == "__main__":
    targets = [
        "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment",
        "Multi-Class Soybean Leaf Disease Dataset Healthy a",
        "SoyCotton",
        "SoyNet Indian Soybean Image dataset with quality i"
    ]
    results = {}
    for t in targets:
        p = RAW_DIR / t
        if p.exists():
            results[t] = audit_dataset_structure(p)
        else:
            print(f"Target not found: {p}")
