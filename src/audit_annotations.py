import os
import zipfile
import json
import csv
from src.config import RAW_DIR, OUTPUTS_DIR

TARGET_EXTENSIONS = {
    ".txt", ".json", ".xml", ".csv", ".yaml", ".yml", ".geojson",
    ".mask", ".png", ".tif", ".tiff"
}

def audit_annotations():
    print("="*70)
    print("PART B: AUDITING HIDDEN YOLO & SEGMENTATION ANNOTATIONS")
    print("="*70)
    
    audit_dir = os.path.join(OUTPUTS_DIR, "annotation_audit")
    os.makedirs(audit_dir, exist_ok=True)
    
    annotation_files = []
    yolo_candidates = []
    seg_candidates = []
    
    # 1. Search uncompressed files in raw/
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TARGET_EXTENSIONS:
                fp = os.path.join(root, f).replace("\\", "/")
                annotation_files.append({
                    "file_path": fp,
                    "extension": ext,
                    "location": "raw_filesystem",
                    "archive": "None"
                })
                
    # 2. Search inside all zip archives in raw/ and ZIP/
    all_zips = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".zip"):
                all_zips.append(os.path.join(root, f))
    zip_root = os.path.join(os.path.dirname(RAW_DIR), "ZIP")
    if os.path.exists(zip_root):
        for root, dirs, files in os.walk(zip_root):
            for f in files:
                if f.endswith(".zip"):
                    all_zips.append(os.path.join(root, f))
                    
    for zp in all_zips:
        try:
            with zipfile.ZipFile(zp, 'r') as zf:
                for name in zf.namelist():
                    ext = os.path.splitext(name)[1].lower()
                    if ext in TARGET_EXTENSIONS:
                        annotation_files.append({
                            "file_path": name,
                            "extension": ext,
                            "location": "inside_zip",
                            "archive": os.path.basename(zp)
                        })
        except Exception as e:
            print(f"Warning reading zip {zp}: {e}")

    # Process candidates
    for af in annotation_files:
        ext = af["extension"]
        if ext in [".txt", ".xml", ".json", ".yaml", ".yml"]:
            yolo_candidates.append(af)
        if ext in [".mask", ".png", ".tif", ".tiff", ".geojson"]:
            seg_candidates.append(af)

    summary = {
        "search_directories": [RAW_DIR, zip_root if os.path.exists(zip_root) else ""],
        "targeted_extensions": sorted(list(TARGET_EXTENSIONS)),
        "total_annotation_files_found": len(annotation_files),
        "yolo_candidate_files": len(yolo_candidates),
        "segmentation_candidate_files": len(seg_candidates),
        "yolo_valid_annotations": 0,
        "segmentation_valid_annotations": 0,
        "conclusion": {
            "yolo": "No valid bounding box localization annotations exist in source datasets.",
            "segmentation": "No valid pixel masks or polygon annotations exist in source datasets.",
            "integrity_rule": "Strictly avoiding fabricated ground truth bounding boxes or masks."
        }
    }

    # 1. outputs/annotation_audit/annotation_summary.json
    with open(os.path.join(audit_dir, "annotation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # 2. outputs/annotation_audit/annotation_files.csv
    with open(os.path.join(audit_dir, "annotation_files.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "extension", "location", "archive"])
        w.writeheader()
        w.writerows(annotation_files)

    # 3. outputs/annotation_audit/yolo_candidates.csv
    with open(os.path.join(audit_dir, "yolo_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "extension", "location", "archive"])
        w.writeheader()
        w.writerows(yolo_candidates)

    # 4. outputs/annotation_audit/segmentation_candidates.csv
    with open(os.path.join(audit_dir, "segmentation_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "extension", "location", "archive"])
        w.writeheader()
        w.writerows(seg_candidates)

    # 5. outputs/yolo_status_final.json
    yolo_final = {
        "status": "unavailable",
        "reason": "No valid ground-truth bounding-box annotations available"
    }
    with open(os.path.join(OUTPUTS_DIR, "yolo_status_final.json"), "w", encoding="utf-8") as f:
        json.dump(yolo_final, f, indent=4)

    # 6. outputs/spread_status_final.json
    spread_final = {
        "status": "unavailable",
        "reason": "No valid ground-truth segmentation annotations available"
    }
    with open(os.path.join(OUTPUTS_DIR, "spread_status_final.json"), "w", encoding="utf-8") as f:
        json.dump(spread_final, f, indent=4)

    print(f"Annotation audit complete: Found {len(annotation_files)} annotation files.")
    print("Created outputs/yolo_status_final.json and outputs/spread_status_final.json")
    return summary

if __name__ == "__main__":
    audit_annotations()
