"""
Crop_AI Deployment — Model Loader
Safe instantiation and loading routines for CNN and YOLO models.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT_DIR = Path("c:/Users/oswal/Music/Crop_AI")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

from src.deployment.model_registry import get_model_entry

def get_cnn_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def build_efficientnet_b0(num_classes: int = 9) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_cnn_classifier(model_path: Path, device: str = "cpu") -> Tuple[nn.Module, list, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model checkpoint not found at {model_path}")
    
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", [])
    num_classes = len(class_names) if class_names else 9
    
    model = build_efficientnet_b0(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    transform = get_cnn_transform()
    return model, class_names, transform

def load_yolo_detector(model_path: Path, device: str = "cpu") -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model weights not found at {model_path}")
    
    model = YOLO(str(model_path))
    return model

def load_all_deployment_models(device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    """
    Loads all four authoritative models into memory for inference.
    """
    print(f"Loading all Crop_AI deployment models on device: {device}")
    
    cnn_entry = get_model_entry("cnn_classifier")
    disease_entry = get_model_entry("disease_detector")
    nutrient_entry = get_model_entry("nutrient_detector")
    leaf_entry = get_model_entry("leaf_detector")
    
    cnn_model, cnn_classes, cnn_transform = load_cnn_classifier(Path(cnn_entry["absolute_path"]), device=device)
    disease_path = Path(disease_entry["absolute_path"])
    print(f"[Model Loader] Authoritative Disease Model Path: {disease_path.resolve()}")
    disease_model = load_yolo_detector(disease_path, device=device)
    print(f"[Model Loader] Disease Model Class Names: {disease_model.names}")
    nutrient_model = load_yolo_detector(Path(nutrient_entry["absolute_path"]), device=device)
    leaf_model = load_yolo_detector(Path(leaf_entry["absolute_path"]), device=device)
    
    return {
        "cnn": {
            "model": cnn_model,
            "classes": cnn_classes,
            "transform": cnn_transform,
            "metadata": cnn_entry
        },
        "disease_yolo": {
            "model": disease_model,
            "classes": disease_model.names,
            "metadata": disease_entry
        },
        "nutrient_yolo": {
            "model": nutrient_model,
            "classes": nutrient_model.names,
            "metadata": nutrient_entry
        },
        "leaf_yolo": {
            "model": leaf_model,
            "classes": leaf_model.names,
            "metadata": leaf_entry
        },
        "device": device
    }
