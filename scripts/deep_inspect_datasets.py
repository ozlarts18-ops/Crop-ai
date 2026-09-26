import os
import json
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("c:/Users/oswal/Music/Crop_AI/raw")

def inspect_mh():
    p = RAW_DIR / "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment"
    print("\n--- IN-DEPTH INSPECTION: MH-SoyaHealthVision ---")
    for item in p.rglob('*'):
        if item.is_file():
            print(f"  File: {item.relative_to(p)} ({item.stat().st_size / (1024*1024):.2f} MB)")
        elif item.is_dir():
            print(f"  Dir:  {item.relative_to(p)}")

def inspect_multiclass():
    p = RAW_DIR / "Multi-Class Soybean Leaf Disease Dataset Healthy a" / "Soyabean leaf desease dataset"
    print("\n--- IN-DEPTH INSPECTION: Multi-Class Soybean Leaf Disease Dataset ---")
    if p.exists():
        for d in sorted(p.iterdir()):
            if d.is_dir():
                imgs = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                print(f"  Class '{d.name}': {len(imgs)} images")
            elif d.is_file():
                print(f"  File in root: {d.name}")

def inspect_soycotton():
    p = RAW_DIR / "SoyCotton"
    print("\n--- IN-DEPTH INSPECTION: SoyCotton ---")
    coco_path = p / "SoyCotton" / "annotations" / "coco.json"
    if coco_path.exists():
        with open(coco_path, 'r') as f:
            coco = json.load(f)
        print("  COCO keys:", list(coco.keys()))
        print("  COCO categories:", coco.get("categories"))
        print("  COCO images count:", len(coco.get("images", [])))
        print("  COCO annotations count:", len(coco.get("annotations", [])))
    for d in (p / "SoyCotton").iterdir():
        if d.is_dir():
            files = list(d.rglob('*'))
            print(f"  Subdir '{d.name}': {len([f for f in files if f.is_file()])} files")

def inspect_soynet():
    p = RAW_DIR / "SoyNet Indian Soybean Image dataset with quality i" / "SoyNet"
    print("\n--- IN-DEPTH INSPECTION: SoyNet ---")
    if p.exists():
        for d in sorted(p.iterdir()):
            if d.is_dir():
                sub_items = list(d.iterdir())
                sub_dirs = [x for x in sub_items if x.is_dir()]
                sub_files = [x for x in sub_items if x.is_file()]
                print(f"  Category '{d.name}': {len(sub_dirs)} subdirs, {len(sub_files)} files directly")
                if sub_dirs:
                    for sd in sorted(sub_dirs)[:10]:
                        imgs = [f for f in sd.iterdir() if f.is_file()]
                        print(f"    - Subclass '{sd.name}': {len(imgs)} images")
                    if len(sub_dirs) > 10:
                        print(f"    ... and {len(sub_dirs) - 10} more subdirectories")

if __name__ == "__main__":
    inspect_mh()
    inspect_multiclass()
    inspect_soycotton()
    inspect_soynet()
