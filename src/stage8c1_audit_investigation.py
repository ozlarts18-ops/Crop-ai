import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from collections import defaultdict
import statistics
import cv2
import numpy as np

ROOT = Path("c:/Users/oswal/Music/Crop_AI")
V2_DIR = ROOT / "data/yolo_disease_v2"
V1_DIR = ROOT / "data/yolo_disease"
CLASSES = ["Charcol rot", "Healthy", "RAB", "Target Leaf Spot"]

def run_investigation():
    splits = ["train", "val", "test"]
    
    # Check all files in V2
    data_by_split = {}
    all_images = []
    
    for s in splits:
        img_dir = V2_DIR / "images" / s
        lbl_dir = V2_DIR / "labels" / s
        
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        print(f"Split {s}: {len(imgs)} images")
        
        split_records = []
        for img_p in imgs:
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            boxes = []
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            boxes.append({"class_id": cid, "class_name": CLASSES[cid], "x": x, "y": y, "w": w, "h": h, "area": w * h})
            
            # Read image metadata
            im = cv2.imread(str(img_p))
            if im is not None:
                h_px, w_px, c_px = im.shape
            else:
                h_px, w_px, c_px = None, None, None
                
            with open(img_p, "rb") as f:
                img_bytes = f.read()
                md5_hash = hashlib.md5(img_bytes).hexdigest()
                sha256_hash = hashlib.sha256(img_bytes).hexdigest()
                
            rec = {
                "split": s,
                "image_path": str(img_p),
                "image_name": img_p.name,
                "stem": img_p.stem,
                "ext": img_p.suffix,
                "width": w_px,
                "height": h_px,
                "boxes": boxes,
                "num_boxes": len(boxes),
                "classes_present": list(set(b["class_id"] for b in boxes)),
                "md5": md5_hash,
                "sha256": sha256_hash
            }
            split_records.append(rec)
            all_images.append(rec)
            
        data_by_split[s] = split_records

    print(f"\nTotal images across splits: {len(all_images)}")
    
    # 1. Split statistics
    print("\n--- SPLIT STATISTICS ---")
    for s in splits:
        recs = data_by_split[s]
        box_counts = [r["num_boxes"] for r in recs]
        print(f"[{s.upper()}] Images: {len(recs)}, Total Boxes: {sum(box_counts)}")
        print(f"  Mean: {statistics.mean(box_counts):.2f}, Median: {statistics.median(box_counts)}, Min: {min(box_counts)}, Max: {max(box_counts)}")
        
        # Per class
        c_counts = defaultdict(int)
        c_img_counts = defaultdict(int)
        for r in recs:
            for cid in set(r["classes_present"]):
                c_img_counts[cid] += 1
            for b in r["boxes"]:
                c_counts[b["class_id"]] += 1
        for cid, cname in enumerate(CLASSES):
            print(f"  {cname} (ID {cid}): {c_img_counts[cid]} images, {c_counts[cid]} boxes")
            
    # 2. Target Leaf Spot Deep Dive
    print("\n--- TARGET LEAF SPOT (TLS) DEEP DIVE ---")
    tls_images = [r for r in all_images if 3 in r["classes_present"]]
    print(f"Total TLS images: {len(tls_images)}")
    for r in tls_images:
        tls_boxes = [b for b in r["boxes"] if b["class_id"] == 3]
        print(f"Split: {r['split']:<5} | Name: {r['image_name']:<40} | TLS boxes: {len(tls_boxes):<4} | Total boxes: {r['num_boxes']:<4} | Res: {r['width']}x{r['height']}")
        
    tls_boxes_by_split = defaultdict(list)
    tls_imgs_by_split = defaultdict(list)
    for r in tls_images:
        tls_boxes = [b for b in r["boxes"] if b["class_id"] == 3]
        tls_boxes_by_split[r['split']].append(len(tls_boxes))
        tls_imgs_by_split[r['split']].append(r)
        
    for s in splits:
        b_list = tls_boxes_by_split[s]
        print(f"TLS in {s.upper()}: {len(tls_imgs_by_split[s])} images, {sum(b_list)} total boxes, box counts per image: {b_list}")

if __name__ == "__main__":
    run_investigation()
