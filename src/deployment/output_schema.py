"""
Crop_AI Deployment — Standardized Output Schema & Validator
Defines the canonical JSON data structure and strict schema validator.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json

# Authoritative Taxonomies
CNN_TAXONOMY = [
    "Bacterial_Blight",
    "Cercospora_Leaf_Blight",
    "Frogeye_Leaf_Spot",
    "Healthy",
    "Mosaic",
    "Pest_Damage",
    "Septoria_Brown_Spot",
    "Soybean_Rust",
    "Sudden_Death_Syndrome"
]

DISEASE_YOLO_TAXONOMY = [
    "Charcol rot",
    "Healthy",
    "RAB",
    "Target Leaf Spot"
]

NUTRIENT_YOLO_TAXONOMY = [
    "Calcium_deficiency",
    "Magnesium_deficiency",
    "N_deficiency",
    "Phosphorus_Deficiency",
    "Potassium_deficiency"
]

FORBIDDEN_FIELDS = [
    "spreadness",
    "spread_percentage",
    "disease_area",
    "severity",
    "affected_area"
]

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> Dict[str, int]:
        return {"x1": int(self.x1), "y1": int(self.y1), "x2": int(self.x2), "y2": int(self.y2)}

@dataclass
class DetectionItem:
    confidence: float
    bounding_box: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(float(self.confidence), 4),
            "bounding_box": self.bounding_box.to_dict()
        }

@dataclass
class DiseaseDetection:
    detected: bool
    disease_type: Optional[str]
    detections: List[DetectionItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": bool(self.detected),
            "disease_type": self.disease_type if self.detected else None,
            "detections": [d.to_dict() for d in self.detections] if self.detected else []
        }

@dataclass
class NutrientDeficiency:
    detected: bool
    type: Optional[str]
    detections: List[DetectionItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": bool(self.detected),
            "type": self.type if self.detected else None,
            "detections": [d.to_dict() for d in self.detections] if self.detected else []
        }

@dataclass
class CropAIPredictionOutput:
    crop: str
    health_status: str
    confidence: float
    disease_detection: DiseaseDetection
    nutrient_deficiency: NutrientDeficiency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop": self.crop,
            "health_status": self.health_status,
            "confidence": round(float(self.confidence), 4),
            "disease_detection": self.disease_detection.to_dict(),
            "nutrient_deficiency": self.nutrient_deficiency.to_dict()
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

def validate_crop_ai_output(data: Any) -> Tuple[bool, str]:
    """
    Validates that a given dictionary strictly complies with the Crop_AI Stage 9 Canonical Schema.
    Returns (True, "Valid") on success or (False, error_message) on failure.
    """
    if not isinstance(data, dict):
        return False, f"Root output must be a dictionary, got {type(data).__name__}"

    # Check for forbidden fields
    def check_forbidden(d: Any, path: str = "") -> Optional[str]:
        if isinstance(d, dict):
            for k, v in d.items():
                curr_path = f"{path}.{k}" if path else k
                if k.lower() in FORBIDDEN_FIELDS:
                    return f"Forbidden field '{curr_path}' detected. Disease spreadness/severity is not supported by bounding boxes."
                res = check_forbidden(v, curr_path)
                if res:
                    return res
        elif isinstance(d, list):
            for i, item in enumerate(d):
                res = check_forbidden(item, f"{path}[{i}]")
                if res:
                    return res
        return None

    forbidden_err = check_forbidden(data)
    if forbidden_err:
        return False, forbidden_err

    # Check required root keys
    required_root_keys = ["crop", "health_status", "confidence", "disease_detection", "nutrient_deficiency"]
    for k in required_root_keys:
        if k not in data:
            return False, f"Missing required root field '{k}'"

    # Disallow unsupported root keys
    for k in data.keys():
        if k not in required_root_keys:
            return False, f"Unsupported root field '{k}' found in payload"

    # 1. Validate 'crop'
    if data["crop"] != "Soybean":
        return False, f"Field 'crop' must be exactly 'Soybean', got '{data['crop']}'"

    # 2. Validate 'health_status'
    if not isinstance(data["health_status"], str):
        return False, f"Field 'health_status' must be a string, got {type(data['health_status']).__name__}"
    if data["health_status"] not in CNN_TAXONOMY:
        return False, f"Field 'health_status' value '{data['health_status']}' is not in the authoritative CNN taxonomy: {CNN_TAXONOMY}"

    # 3. Validate 'confidence'
    conf = data["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        return False, f"Field 'confidence' must be numeric float, got {type(conf).__name__}"
    if not (0.0 <= conf <= 1.0):
        return False, f"Field 'confidence' must be between 0.0 and 1.0, got {conf}"

    # 4. Validate 'disease_detection'
    dd = data["disease_detection"]
    if not isinstance(dd, dict):
        return False, f"Field 'disease_detection' must be a dictionary, got {type(dd).__name__}"
    for k in ["detected", "disease_type", "detections"]:
        if k not in dd:
            return False, f"Missing field '{k}' in 'disease_detection'"
    for k in dd.keys():
        if k not in ["detected", "disease_type", "detections"]:
            return False, f"Unsupported field '{k}' in 'disease_detection'"

    if not isinstance(dd["detected"], bool):
        return False, f"'disease_detection.detected' must be a boolean, got {type(dd['detected']).__name__}"

    if dd["detected"]:
        if not isinstance(dd["disease_type"], str) or not dd["disease_type"].strip():
            return False, "'disease_detection.disease_type' must be a non-empty string when detected is true"
        if not isinstance(dd["detections"], list) or len(dd["detections"]) == 0:
            return False, "'disease_detection.detections' must be a non-empty list when detected is true"
    else:
        if dd["disease_type"] is not None:
            return False, f"'disease_detection.disease_type' must be null when detected is false, got '{dd['disease_type']}'"
        if not isinstance(dd["detections"], list) or len(dd["detections"]) != 0:
            return False, "'disease_detection.detections' must be an empty list [] when detected is false"

    # Validate detection items in disease_detection
    for idx, det in enumerate(dd["detections"]):
        val_res, val_msg = _validate_detection_item(det, f"disease_detection.detections[{idx}]")
        if not val_res:
            return False, val_msg

    # 5. Validate 'nutrient_deficiency'
    nd = data["nutrient_deficiency"]
    if not isinstance(nd, dict):
        return False, f"Field 'nutrient_deficiency' must be a dictionary, got {type(nd).__name__}"
    for k in ["detected", "type", "detections"]:
        if k not in nd:
            return False, f"Missing field '{k}' in 'nutrient_deficiency'"
    for k in nd.keys():
        if k not in ["detected", "type", "detections"]:
            return False, f"Unsupported field '{k}' in 'nutrient_deficiency'"

    if not isinstance(nd["detected"], bool):
        return False, f"'nutrient_deficiency.detected' must be a boolean, got {type(nd['detected']).__name__}"

    if nd["detected"]:
        if not isinstance(nd["type"], str) or not nd["type"].strip():
            return False, "'nutrient_deficiency.type' must be a non-empty string when detected is true"
        if not isinstance(nd["detections"], list) or len(nd["detections"]) == 0:
            return False, "'nutrient_deficiency.detections' must be a non-empty list when detected is true"
    else:
        if nd["type"] is not None:
            return False, f"'nutrient_deficiency.type' must be null when detected is false, got '{nd['type']}'"
        if not isinstance(nd["detections"], list) or len(nd["detections"]) != 0:
            return False, "'nutrient_deficiency.detections' must be an empty list [] when detected is false"

    # Validate detection items in nutrient_deficiency
    for idx, det in enumerate(nd["detections"]):
        val_res, val_msg = _validate_detection_item(det, f"nutrient_deficiency.detections[{idx}]")
        if not val_res:
            return False, val_msg

    return True, "Valid"

def _validate_detection_item(item: Any, path: str) -> Tuple[bool, str]:
    if not isinstance(item, dict):
        return False, f"Detection item at '{path}' must be a dictionary"
    for k in ["confidence", "bounding_box"]:
        if k not in item:
            return False, f"Missing '{k}' in detection item at '{path}'"
    for k in item.keys():
        if k not in ["confidence", "bounding_box"]:
            return False, f"Unsupported field '{k}' in detection item at '{path}'"

    c = item["confidence"]
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return False, f"Confidence at '{path}' must be a numeric float, got {type(c).__name__}"
    if not (0.0 <= c <= 1.0):
        return False, f"Confidence at '{path}' must be between 0.0 and 1.0, got {c}"

    bb = item["bounding_box"]
    if not isinstance(bb, dict):
        return False, f"Bounding box at '{path}' must be a dictionary"
    for k in ["x1", "y1", "x2", "y2"]:
        if k not in bb:
            return False, f"Missing '{k}' in bounding box at '{path}'"
        v = bb[k]
        if not isinstance(v, int) or isinstance(v, bool):
            return False, f"Coordinate '{k}' at '{path}' must be an integer, got {type(v).__name__}"
        if v < 0:
            return False, f"Coordinate '{k}' at '{path}' cannot be negative, got {v}"

    if bb["x2"] < bb["x1"]:
        return False, f"Invalid bounding box at '{path}': x2 ({bb['x2']}) is less than x1 ({bb['x1']})"
    if bb["y2"] < bb["y1"]:
        return False, f"Invalid bounding box at '{path}': y2 ({bb['y2']}) is less than y1 ({bb['y1']})"

    return True, "Valid"
