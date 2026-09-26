import zipfile
import os
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("c:/Users/oswal/Music/Crop_AI/raw")

def probe_mh_zips():
    p = RAW_DIR / "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment"
    print("\n==========================================")
    print("PROBING MH-SOYAHEALTHVISION ZIP FILES")
    print("==========================================")
    total_imgs = 0
    total_annotations = 0
    for zpath in sorted(p.rglob('*.zip')):
        rel = zpath.relative_to(p)
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                infolist = zf.infolist()
                files = [f for f in infolist if not f.is_dir()]
                exts = defaultdict(int)
                for f in files:
                    ext = Path(f.filename).suffix.lower()
                    exts[ext] += 1
                img_count = sum(exts[e] for e in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'])
                ann_count = sum(exts[e] for e in ['.json', '.xml', '.txt', '.csv', '.yaml', '.yml'])
                total_imgs += img_count
                total_annotations += ann_count
                print(f"{rel}:")
                print(f"  Total items: {len(files)} | Images: {img_count} | Annotations: {ann_count}")
                print(f"  Extensions: {dict(exts)}")
                # Check sample paths
                sample_paths = [f.filename for f in files[:3]]
                print(f"  Sample contents: {sample_paths}")
        except Exception as e:
            print(f"Error reading {rel}: {e}")
    print(f"\nMH-SoyaHealthVision GRAND TOTAL across all zips: {total_imgs} images, {total_annotations} annotations")

def probe_soynet_deep():
    p = RAW_DIR / "SoyNet Indian Soybean Image dataset with quality i" / "SoyNet"
    print("\n==========================================")
    print("PROBING SOYNET DIRECTORY TREE DEEPLY")
    print("==========================================")
    all_files = list(p.rglob('*'))
    img_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"Total files: {len(all_files)}, Image files: {len(img_files)}")
    
    # Analyze directory structures
    leaf_dirs = defaultdict(int)
    for img in img_files:
        rel = img.relative_to(p)
        parent = str(rel.parent)
        leaf_dirs[parent] += 1
        
    print(f"\nUnique image subdirectories ({len(leaf_dirs)}):")
    for parent, count in sorted(leaf_dirs.items()):
        print(f"  {parent}: {count} images")

if __name__ == "__main__":
    probe_mh_zips()
    probe_soynet_deep()
