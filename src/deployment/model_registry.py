"""
Crop_AI Deployment — Authoritative Model Registry
Contains the catalog and metadata of all 4 trained, frozen models.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cnn_classifier": {
        "model_id": "MOD_CNN_01",
        "name": "EfficientNet-B0 Health Classifier",
        "relative_path": "models/optimized/best_model.pt",
        "absolute_path": str(ROOT_DIR / "models/optimized/best_model.pt"),
        "model_type": "classification",
        "architecture": "EfficientNet-B0",
        "authoritative_stage": "Stage 8 Optimization",
        "version": "8.0.0-opt",
        "input_resolution": [224, 224],
        "classes": [
            "Bacterial_Blight",
            "Cercospora_Leaf_Blight",
            "Frogeye_Leaf_Spot",
            "Healthy",
            "Mosaic",
            "Pest_Damage",
            "Septoria_Brown_Spot",
            "Soybean_Rust",
            "Sudden_Death_Syndrome"
        ],
        "num_classes": 9,
        "primary_responsibility": "Provides the canonical 'health_status' field and overall classification confidence score",
        "sha256": "e978438a05fca1ee30d33c49576e321bc132a06d939ed6a4463b44676f7b43c3",
        "size_mb": 46.41
    },
    "disease_detector": {
        "model_id": "MOD_YOLO_DISEASE_V3",
        "name": "YOLO11m Disease Region Detector",
        "relative_path": "models/yolo11m_disease_v3_100/best.pt",
        "absolute_path": str(ROOT_DIR / "models/yolo11m_disease_v3_100/best.pt"),
        "model_type": "object_detection",
        "architecture": "YOLO11m",
        "authoritative_stage": "Stage 8D Clean V3 100-Epoch",
        "version": "8.4.0-v3-100ep",
        "input_resolution": [640, 640],
        "classes": [
            "Charcol rot",
            "Healthy",
            "RAB",
            "Target Leaf Spot"
        ],
        "num_classes": 4,
        "primary_responsibility": "Localizes disease lesion bounding boxes ('disease_detection' block). Independent of CNN taxonomy.",
        "sha256": "469698c536e3450df24f6469500f9982ca932d350b924ec87f5eb46d5a577b8c",
        "size_mb": 38.62
    },
    "nutrient_detector": {
        "model_id": "MOD_YOLO_NUTRIENT_01",
        "name": "YOLO11m Nutrient Deficiency Detector",
        "relative_path": "models/yolo11m_nutrient_optimized/best.pt",
        "absolute_path": str(ROOT_DIR / "models/yolo11m_nutrient_optimized/best.pt"),
        "model_type": "object_detection",
        "architecture": "YOLO11m",
        "authoritative_stage": "Stage 8B Controlled Optimization",
        "version": "8.2.0-opt",
        "input_resolution": [640, 640],
        "classes": [
            "Calcium_deficiency",
            "Magnesium_deficiency",
            "N_deficiency",
            "Phosphorus_Deficiency",
            "Potassium_deficiency"
        ],
        "num_classes": 5,
        "primary_responsibility": "Localizes nutrient deficiency symptom bounding boxes ('nutrient_deficiency' block)",
        "sha256": "ed9e003c8d8e25c00a29b0aca72d7adff099a87618e86fe94ba5154b2123c18f",
        "size_mb": 38.64
    },
    "leaf_detector": {
        "model_id": "MOD_YOLO_LEAF_01",
        "name": "YOLO11m Soybean Leaf Detector",
        "relative_path": "models/yolo11m/best.pt",
        "absolute_path": str(ROOT_DIR / "models/yolo11m/best.pt"),
        "model_type": "object_detection",
        "architecture": "YOLO11m",
        "authoritative_stage": "SoyCotton Leaf Pretraining",
        "version": "6.0.0",
        "input_resolution": [640, 640],
        "classes": [
            "soybean_leaf"
        ],
        "num_classes": 1,
        "primary_responsibility": "Localizes soybean leaves within complex canopy scenes. Never conflated with disease.",
        "sha256": "24e5285fb0cc932e954fe80c67939cd997223cde5d61b167ce1a24a2b5dae7b2",
        "size_mb": 38.63
    }
}

def get_registry() -> Dict[str, Dict[str, Any]]:
    return MODEL_REGISTRY

def get_model_entry(model_key: str) -> Dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Model key '{model_key}' not found in registry. Available keys: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def compute_file_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
