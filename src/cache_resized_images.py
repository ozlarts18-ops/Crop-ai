"""
cache_resized_images.py
Multi-threaded pre-resizing of high-resolution images to 256x256 in data/prepared/
to eliminate JPEG decoding bottleneck during training.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
DATA_DIR = ROOT_DIR / "data"
META_DIR = DATA_DIR / "metadata"
PREP_DIR = DATA_DIR / "prepared"


def process_single_image(args):
    orig_p_str, h, cache_dir_str = args
    orig_p = Path(orig_p_str)
    cached_p = Path(cache_dir_str) / f"{h[:12]}_{orig_p.name}"

    if not cached_p.exists():
        try:
            with Image.open(orig_p) as im:
                im = im.convert("RGB")
                im = im.resize((256, 256), Image.Resampling.BILINEAR)
                im.save(cached_p, quality=90)
        except Exception as e:
            return str(orig_p.resolve())

    return str(cached_p.resolve())


def cache_manifest(csv_name, cache_subfolder):
    csv_path = META_DIR / csv_name
    df = pd.read_csv(csv_path)
    cache_dir = PREP_DIR / cache_subfolder
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching {len(df)} images from {csv_name} into {cache_subfolder} using ThreadPool...")
    args_list = [(row["filepath"], row["hash_md5"], str(cache_dir.resolve())) for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=12) as pool:
        new_filepaths = list(pool.map(process_single_image, args_list))

    df["filepath"] = new_filepaths
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(new_filepaths)} cached paths.\n")


if __name__ == "__main__":
    cache_manifest("disease_metadata.csv", "disease_cached_256")
    cache_manifest("nutrient_metadata.csv", "nutrient_cached_256")
    cache_manifest("crop_verification_metadata.csv", "crop_verifier_cached_256")
    print("Multi-threaded pre-caching complete!")
