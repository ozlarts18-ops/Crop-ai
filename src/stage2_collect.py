import os
import zipfile
import json
import csv
from src.config import RAW_DIR, OUTPUTS_DIR

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"

def run_stage2():
    print("Executing Stage 2: Collect Data (Inspecting raw/)...")
    
    datasets_info = {}
    structure_lines = []
    
    # 1. MH-SoyaHealthVision
    mh_dir = os.path.join(RAW_DIR, "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment")
    mh_size = 0
    mh_img_count = 0
    mh_classes = set()
    mh_structure = {}
    
    if os.path.exists(mh_dir):
        for root, dirs, files in os.walk(mh_dir):
            for f in files:
                fp = os.path.join(root, f)
                f_size = os.path.getsize(fp)
                mh_size += f_size
                if f.endswith('.zip'):
                    rel_dir = os.path.relpath(root, mh_dir)
                    if rel_dir not in mh_structure:
                        mh_structure[rel_dir] = []
                    
                    with zipfile.ZipFile(fp, 'r') as zf:
                        namelist = zf.namelist()
                        img_files = [n for n in namelist if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        mh_img_count += len(img_files)
                        # Extract class name from zip or inner folder
                        class_name = os.path.splitext(f)[0]
                        mh_classes.add(class_name)
                        mh_structure[rel_dir].append({
                            "zip_name": f,
                            "size_bytes": f_size,
                            "image_count": len(img_files),
                            "sample_images": [os.path.basename(n) for n in img_files[:3]]
                        })

    datasets_info["MH-SoyaHealthVision"] = {
        "dataset_name": "MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment",
        "total_images": mh_img_count,
        "size_bytes": mh_size,
        "size_human": format_size(mh_size),
        "classes": sorted(list(mh_classes)),
        "annotation_types": ["Folder-level classification"],
        "bounding_boxes": 0,
        "segmentation_masks": 0,
        "classification_labels": mh_img_count,
        "existing_splits": "None (unsplit)",
        "missing_annotations": "No bounding boxes or pixel masks present",
        "subsets": mh_structure
    }

    # 2. Multi-Class Soybean Leaf Disease Dataset
    mc_dir = os.path.join(RAW_DIR, "Multi-Class Soybean Leaf Disease Dataset Healthy a")
    mc_size = 0
    mc_img_count = 0
    mc_classes = {}
    
    if os.path.exists(mc_dir):
        for root, dirs, files in os.walk(mc_dir):
            for f in files:
                fp = os.path.join(root, f)
                f_size = os.path.getsize(fp)
                mc_size += f_size
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    mc_img_count += 1
                    class_folder = os.path.basename(root)
                    mc_classes[class_folder] = mc_classes.get(class_folder, 0) + 1

    datasets_info["Multi-Class Soybean Leaf Disease Dataset"] = {
        "dataset_name": "Multi-Class Soybean Leaf Disease Dataset Healthy a",
        "total_images": mc_img_count,
        "size_bytes": mc_size,
        "size_human": format_size(mc_size),
        "classes": sorted(list(mc_classes.keys())),
        "class_breakdown": mc_classes,
        "annotation_types": ["Folder-level classification"],
        "bounding_boxes": 0,
        "segmentation_masks": 0,
        "classification_labels": mc_img_count,
        "existing_splits": "None (unsplit)",
        "missing_annotations": "No bounding boxes or pixel masks present"
    }

    # 3. SoyNet
    sn_dir = os.path.join(RAW_DIR, "SoyNet Indian Soybean Image dataset with quality i")
    sn_size = 0
    sn_img_count = 0
    sn_folders = {}
    
    if os.path.exists(sn_dir):
        for root, dirs, files in os.walk(sn_dir):
            for f in files:
                fp = os.path.join(root, f)
                f_size = os.path.getsize(fp)
                sn_size += f_size
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sn_img_count += 1
                    rel_dir = os.path.relpath(root, sn_dir)
                    sn_folders[rel_dir] = sn_folders.get(rel_dir, 0) + 1

    datasets_info["SoyNet"] = {
        "dataset_name": "SoyNet Indian Soybean Image dataset with quality i",
        "total_images": sn_img_count,
        "size_bytes": sn_size,
        "size_human": format_size(sn_size),
        "classes": ["Healthy", "Disease", "Mobile pic (Unspecified)"],
        "folder_distribution": sn_folders,
        "annotation_types": ["Folder-level binary/unspecified classification"],
        "bounding_boxes": 0,
        "segmentation_masks": 0,
        "classification_labels": sn_img_count,
        "existing_splits": "None (unsplit)",
        "missing_annotations": "No multi-class disease labels; no bounding boxes or pixel masks"
    }

    # Build structure text
    structure_lines.append("="*80)
    structure_lines.append("DATASET STRUCTURE & INVENTORY REPORT")
    structure_lines.append("="*80)
    for ds_key, ds_val in datasets_info.items():
        structure_lines.append(f"\n[DATASET]: {ds_key}")
        structure_lines.append(f"  Full Name: {ds_val['dataset_name']}")
        structure_lines.append(f"  Total Images: {ds_val['total_images']}")
        structure_lines.append(f"  Total Size: {ds_val['size_human']} ({ds_val['size_bytes']} bytes)")
        structure_lines.append(f"  Classes Identified: {ds_val['classes']}")
        structure_lines.append(f"  Annotation Types: {', '.join(ds_val['annotation_types'])}")
        structure_lines.append(f"  Bounding Boxes: {ds_val['bounding_boxes']}")
        structure_lines.append(f"  Segmentation Masks: {ds_val['segmentation_masks']}")
        structure_lines.append(f"  Existing Splits: {ds_val['existing_splits']}")
        structure_lines.append(f"  Missing Annotations: {ds_val['missing_annotations']}")

    # Write outputs/data_collection_report.json
    json_path = os.path.join(OUTPUTS_DIR, "data_collection_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(datasets_info, f, indent=4)
        
    # Write outputs/dataset_structure.txt
    txt_path = os.path.join(OUTPUTS_DIR, "dataset_structure.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(structure_lines))

    # Write outputs/data_collection_report.csv
    csv_path = os.path.join(OUTPUTS_DIR, "data_collection_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_key", "dataset_name", "total_images", "size_bytes", "size_human",
            "classes", "annotation_types", "bounding_boxes", "segmentation_masks",
            "classification_labels", "existing_splits", "missing_annotations"
        ])
        for k, v in datasets_info.items():
            writer.writerow([
                k, v["dataset_name"], v["total_images"], v["size_bytes"], v["size_human"],
                "; ".join(v["classes"]), "; ".join(v["annotation_types"]),
                v["bounding_boxes"], v["segmentation_masks"], v["classification_labels"],
                v["existing_splits"], v["missing_annotations"]
            ])

    print(f"Stage 2 complete: Saved {json_path}, {csv_path}, and {txt_path}")
    return datasets_info

if __name__ == "__main__":
    run_stage2()
