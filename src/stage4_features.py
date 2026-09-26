import os
import json
import csv
import shutil
from src.config import (
    PROCESSED_DIR, YOLO_DIR, CNN_DIR, SEG_DIR, METADATA_DIR,
    PLANT_NAME, SCIENTIFIC_NAME
)

def run_stage4():
    print("Executing Stage 4: Identify Features & Target (Building Feature Structure)...")
    
    # Ensure required directories exist
    os.makedirs(YOLO_DIR, exist_ok=True)
    os.makedirs(CNN_DIR, exist_ok=True)
    os.makedirs(SEG_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Subdirectories for YOLO
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        os.makedirs(os.path.join(YOLO_DIR, sub), exist_ok=True)
        
    # Subdirectories for CNN
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(CNN_DIR, split), exist_ok=True)
        
    # Subdirectories for Segmentation
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SEG_DIR, split), exist_ok=True)
        
    # Read clean records from Stage 3
    clean_records_path = os.path.join(PROCESSED_DIR, "clean_records.json")
    if not os.path.exists(clean_records_path):
        raise FileNotFoundError(f"Clean records not found at {clean_records_path}. Run Stage 3 first.")
        
    with open(clean_records_path, "r", encoding="utf-8") as f:
        clean_records = json.load(f)
        
    print(f"Loaded {len(clean_records)} clean records for feature setup.")
    print("Stage 4 setup complete: Directories initialized.")
    return clean_records

if __name__ == "__main__":
    run_stage4()
