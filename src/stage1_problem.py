import os
import json
from src.config import DOCS_DIR, PLANT_NAME, SCIENTIFIC_NAME, NORMALIZED_CLASSES

def run_stage1():
    print("Executing Stage 1: Real-World Problem Definition...")
    problem_doc = {
        "project_name": "Crop_AI",
        "domain": "Agricultural Artificial Intelligence & Smart Farming",
        "crop_details": {
            "plant_name": PLANT_NAME,
            "scientific_name": SCIENTIFIC_NAME,
            "taxonomy_family": "Fabaceae",
            "economic_importance": "Critical oilseed and protein crop globally, highly vulnerable to fungal, bacterial, and viral foliar diseases."
        },
        "system_objectives": [
            "Plant name identification: Soybean",
            "Scientific name identification: Glycine max",
            "Health status discrimination (Healthy vs. Diseased)",
            "Pathogen/condition classification among normalized classes",
            "Disease confidence estimation",
            "Disease localization (bounding boxes when annotations are available)",
            "Affected area quantification (pixels and spread percentage where segmentation is available)",
            "Severity classification based on lesion coverage"
        ],
        "target_disease_taxonomy": NORMALIZED_CLASSES,
        "operational_constraints": [
            "Pure software machine learning solution (no edge hardware/IoT dependencies)",
            "Strict integrity preservation: no fabricated annotations or bounding boxes",
            "Robust deduplication and data leakage prevention using hashing across train/val/test splits"
        ]
    }
    
    doc_path = os.path.join(DOCS_DIR, "stage1_problem_definition.json")
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(problem_doc, f, indent=4)
        
    md_path = os.path.join(DOCS_DIR, "stage1_problem_definition.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 1: Real-World Problem Definition - {PLANT_NAME} ({SCIENTIFIC_NAME})\n\n")
        f.write("## 1. Problem Overview\n")
        f.write("Early, accurate diagnosis of soybean foliar diseases and pest attacks is vital for food security and minimizing yield losses.\n\n")
        f.write("## 2. Core Detection Targets\n")
        f.write(f"- **Plant Name:** {PLANT_NAME}\n")
        f.write(f"- **Scientific Name:** {SCIENTIFIC_NAME}\n")
        f.write("- **Health Status:** Healthy / Diseased\n")
        f.write("- **Disease Classification:** Identification across 11 normalized pathological classes\n")
        f.write("- **Confidence Scoring:** Calibrated probability distribution\n")
        f.write("- **Spatial Localization:** Object detection coordinates (where localization data exists)\n")
        f.write("- **Severity & Spreadness:** Disease area ratio / spread percentage (where segmentation data exists)\n\n")
        f.write("## 3. Normalized Disease Classes\n")
        for k, v in NORMALIZED_CLASSES.items():
            f.write(f"- `{k}`: {v}\n")
            
    print(f"Stage 1 complete: Saved documentation to {doc_path} and {md_path}")
    return problem_doc

if __name__ == "__main__":
    run_stage1()
