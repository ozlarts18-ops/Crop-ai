"""
audit_farmbot_dataset.py
Audits FarmBot images, YOLO annotations, bounding boxes, duplicates, and instance counts.
Generates data/metadata/farmbot_manifest.csv
"""

import os
import glob
import hashlib
from PIL import Image
import pandas as pd

base = 'data/prepared/growth_yolo'

stats = {
    'train': {'images': 0, 'plant': 0, 'weed': 0},
    'val': {'images': 0, 'plant': 0, 'weed': 0},
    'test': {'images': 0, 'plant': 0, 'weed': 0}
}

corrupt_images = 0
invalid_boxes = 0
unique_hashes = set()
duplicates = 0
manifest_rows = []

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(base, 'images', split)
    lbl_dir = os.path.join(base, 'labels', split)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    stats[split]['images'] = len(img_files)
    
    for img_p in img_files:
        fname = os.path.basename(img_p)
        lbl_p = os.path.join(lbl_dir, os.path.splitext(fname)[0] + '.txt')
        
        try:
            with Image.open(img_p) as im:
                w, h = im.size
            with open(img_p, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
                if md5 in unique_hashes:
                    duplicates += 1
                else:
                    unique_hashes.add(md5)
        except Exception:
            corrupt_images += 1
            continue
            
        plant_count = 0
        weed_count = 0
        if os.path.exists(lbl_p):
            with open(lbl_p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    if cls_id not in (0, 1) or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        invalid_boxes += 1
                    if cls_id == 0:
                        plant_count += 1
                    elif cls_id == 1:
                        weed_count += 1
                        
        stats[split]['plant'] += plant_count
        stats[split]['weed'] += weed_count
        
        manifest_rows.append({
            'filename': fname,
            'split': split,
            'img_path': os.path.abspath(img_p),
            'lbl_path': os.path.abspath(lbl_p),
            'width': w,
            'height': h,
            'hash_md5': md5,
            'plant_instances': plant_count,
            'weed_instances': weed_count
        })

df_manifest = pd.DataFrame(manifest_rows)
os.makedirs('data/metadata', exist_ok=True)
df_manifest.to_csv('data/metadata/farmbot_manifest.csv', index=False)

# Also update data/splits/farmbot_*.csv
os.makedirs('data/splits', exist_ok=True)
for split in ['train', 'val', 'test']:
    sub_df = df_manifest[df_manifest['split'] == split]
    sub_df.to_csv(f'data/splits/farmbot_{split}.csv', index=False)

tot_img = sum(s['images'] for s in stats.values())
tot_p = sum(s['plant'] for s in stats.values())
tot_w = sum(s['weed'] for s in stats.values())

print("==================================================")
print("FARMBOT DATASET AUDIT STATISTICS")
print("==================================================")
print(f"Total images: {tot_img}")
print(f"Train images: {stats['train']['images']}")
print(f"Validation images: {stats['val']['images']}")
print(f"Test images: {stats['test']['images']}")
print("")
print(f"Total plant instances: {tot_p}")
print(f"Total weed instances: {tot_w}")
print("")
print(f"Train plant instances: {stats['train']['plant']}")
print(f"Train weed instances: {stats['train']['weed']}")
print("")
print(f"Validation plant instances: {stats['val']['plant']}")
print(f"Validation weed instances: {stats['val']['weed']}")
print("")
print(f"Test plant instances: {stats['test']['plant']}")
print(f"Test weed instances: {stats['test']['weed']}")
print("")
print(f"Corrupted images: {corrupt_images}")
print(f"Invalid bounding boxes: {invalid_boxes}")
print(f"Exact duplicate images: {duplicates}")
print(f"Unique MD5 hashes: {len(unique_hashes)}")
print("==================================================")
