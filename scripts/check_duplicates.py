import os
import hashlib
import zipfile
from pathlib import Path
from collections import defaultdict
from PIL import Image

RAW_DIR = Path("c:/Users/oswal/Music/Crop_AI/raw")

def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def check_duplicates():
    print("\n==========================================")
    print("CHECKING DUPLICATES ACROSS DATASETS")
    print("==========================================")
    
    hash_to_sources = defaultdict(list)
    
    # 1. Multi-Class
    mc_dir = RAW_DIR / "Multi-Class Soybean Leaf Disease Dataset Healthy a" / "Soyabean leaf desease dataset"
    print("Hashing Multi-Class Dataset...")
    for f in mc_dir.rglob('*.jpg'):
        h = get_file_hash(f.read_bytes())
        hash_to_sources[h].append(("Multi-Class", f.name, str(f.parent.name)))
        
    # 2. SoyCotton
    sc_dir = RAW_DIR / "SoyCotton" / "SoyCotton" / "images"
    print("Hashing SoyCotton Dataset...")
    if sc_dir.exists():
        for f in sc_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']:
                h = get_file_hash(f.read_bytes())
                hash_to_sources[h].append(("SoyCotton", f.name, "images"))

    # 3. SoyNet Raw
    sn_dir = RAW_DIR / "SoyNet Indian Soybean Image dataset with quality i" / "SoyNet" / "Raw_SoyNet_Data"
    print("Hashing SoyNet Raw Dataset...")
    if sn_dir.exists():
        for f in sn_dir.rglob('*.jpg'):
            h = get_file_hash(f.read_bytes())
            hash_to_sources[h].append(("SoyNet_Raw", f.name, str(f.parent.name)))

    # 4. MH-SoyaHealthVision Zips
    mh_dir = RAW_DIR / "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment"
    print("Hashing MH-SoyaHealthVision Zip Archives...")
    for zpath in mh_dir.rglob('*.zip'):
        cat = zpath.stem
        with zipfile.ZipFile(zpath, 'r') as zf:
            for info in zf.infolist():
                if not info.is_dir() and info.filename.lower().endswith(('.jpg', '.jpeg')):
                    data = zf.read(info.filename)
                    h = get_file_hash(data)
                    hash_to_sources[h].append(("MH-SoyaHealthVision", Path(info.filename).name, cat))

    print(f"\nTotal unique file hashes across audited datasets: {len(hash_to_sources)}")
    
    # Analyze exact duplicates
    internal_dupes = defaultdict(list)
    cross_dataset_dupes = defaultdict(list)
    
    for h, sources in hash_to_sources.items():
        if len(sources) > 1:
            datasets = set(s[0] for s in sources)
            if len(datasets) > 1:
                cross_dataset_dupes[tuple(sorted(datasets))].append((sources))
            else:
                internal_dupes[list(datasets)[0]].append(sources)
                
    print("\n--- INTERNAL DUPLICATES WITHIN SAME DATASET ---")
    for ds, dupes in internal_dupes.items():
        print(f"Dataset '{ds}': {len(dupes)} sets of duplicate images (total duplicates: {sum(len(x)-1 for x in dupes)})")
        for s in dupes[:3]:
            print(f"   Duplicate set: {[f'{x[1]} in {x[2]}' for x in s]}")
            
    print("\n--- CROSS-DATASET DUPLICATES ---")
    for ds_pair, dupes in cross_dataset_dupes.items():
        print(f"Datasets {ds_pair}: {len(dupes)} identical images shared between datasets!")
        for s in dupes[:5]:
            print(f"   Shared image: {[f'{x[0]}: {x[1]} ({x[2]})' for x in s]}")

if __name__ == "__main__":
    check_duplicates()
