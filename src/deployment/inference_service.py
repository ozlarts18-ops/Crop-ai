"""
Crop_AI Deployment — Authoritative ONNX Inference Service
Assembles the 4 trained AI models via ONNX Runtime:
1. Crop Verification (EfficientNetV2-S ONNX)
2. Disease Classification (EfficientNetV2-M ONNX)
3. Potassium Deficiency (EfficientNetV2-S ONNX)
4. Plant / Weed Detection (YOLO11-M ONNX)

Models are loaded once at startup and sessions are reused across all requests.
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def format_display_name(raw_name: str) -> str:
    """Converts raw model class identifiers (e.g. Sudden_Death_Syndrome) into clean readable names."""
    if not raw_name:
        return "None"
    name_map = {
        "Normal_Status": "Normal",
        "Potassium_Deficiency": "Potassium Deficiency",
        "Bacterial_Blight": "Bacterial Blight",
        "Cercospora_Leaf_Blight": "Cercospora Leaf Blight",
        "Frogeye_Leaf_Spot": "Frogeye Leaf Spot",
        "Healthy": "Healthy",
        "Mosaic": "Mosaic",
        "Pest_Damage": "Pest Damage",
        "Septoria_Brown_Spot": "Septoria Brown Spot",
        "Soybean_Rust": "Soybean Rust",
        "Sudden_Death_Syndrome": "Sudden Death Syndrome",
        "plant": "Plant",
        "weed": "Weed"
    }
    return name_map.get(raw_name, raw_name.replace("_", " "))


class CropAIInferenceService:
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(models_dir) if models_dir else ROOT_DIR
        self.exports_dir = self.root_dir / "models" / "exports"
        self.labels_dir = self.root_dir / "models" / "labels"

        print("Initializing ONNX Inference Sessions...")

        # 1. Load Crop Verification Model & Threshold
        crop_model_path = self.exports_dir / "crop_verification_efficientnetv2_s.onnx"
        if not crop_model_path.exists():
            raise FileNotFoundError(f"Crop verification model not found at {crop_model_path}")
        self.crop_session = ort.InferenceSession(str(crop_model_path))
        self.crop_input_name = self.crop_session.get_inputs()[0].name

        crop_thresh_path = self.labels_dir / "crop_verification_threshold.json"
        with open(crop_thresh_path, "r") as f:
            crop_meta = json.load(f)
            self.crop_threshold = float(crop_meta.get("selected_threshold", 0.35))
        print(f"Loaded Crop Verifier (Threshold: {self.crop_threshold})")

        # 2. Load Disease Classification Model & Labels
        disease_model_path = self.exports_dir / "disease_efficientnetv2_m.onnx"
        if not disease_model_path.exists():
            raise FileNotFoundError(f"Disease model not found at {disease_model_path}")
        self.disease_session = ort.InferenceSession(str(disease_model_path))
        self.disease_input_name = self.disease_session.get_inputs()[0].name

        disease_labels_path = self.labels_dir / "disease_labels.json"
        with open(disease_labels_path, "r") as f:
            raw_labels = json.load(f)
            self.disease_labels = {int(k): v for k, v in raw_labels.items()}
        print(f"Loaded Disease Classifier ({len(self.disease_labels)} classes)")

        # 3. Load Potassium Model, Labels & Threshold
        potassium_model_path = self.exports_dir / "potassium_efficientnetv2_s.onnx"
        if not potassium_model_path.exists():
            raise FileNotFoundError(f"Potassium model not found at {potassium_model_path}")
        self.potassium_session = ort.InferenceSession(str(potassium_model_path))
        self.potassium_input_name = self.potassium_session.get_inputs()[0].name

        pot_labels_path = self.labels_dir / "potassium_labels.json"
        with open(pot_labels_path, "r") as f:
            raw_pot_labels = json.load(f)
            self.potassium_labels = {int(k): v for k, v in raw_pot_labels.items()}

        pot_thresh_path = self.labels_dir / "potassium_threshold.json"
        with open(pot_thresh_path, "r") as f:
            pot_meta = json.load(f)
            self.potassium_threshold = float(pot_meta.get("selected_threshold", 0.45))
        print(f"Loaded Potassium Model (Threshold: {self.potassium_threshold})")

        # 4. Load Plant / Weed YOLO Model
        yolo_model_path = self.exports_dir / "plant_weed_yolo11_m.onnx"
        if not yolo_model_path.exists():
            raise FileNotFoundError(f"Plant/weed YOLO model not found at {yolo_model_path}")
        self.yolo_session = ort.InferenceSession(str(yolo_model_path))
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name

        plant_weed_labels_path = self.labels_dir / "plant_weed_labels.json"
        with open(plant_weed_labels_path, "r") as f:
            raw_pw_labels = json.load(f)
            self.plant_weed_labels = {int(k): v for k, v in raw_pw_labels.items()}
        print("Loaded Plant / Weed YOLO11-M Model")

        # Standard ImageNet normalization parameters for EfficientNet models
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Standardizes input image to RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Input image not found at {p}")
            return Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return Image.fromarray(image_input).convert("RGB")
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _preprocess_224(self, image: Image.Image) -> np.ndarray:
        """Resizes to 224x224 and applies ImageNet normalization for EfficientNet models."""
        img_resized = image.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        # Transpose HWC -> CHW, add batch dimension -> (1, 3, 224, 224)
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)
        return tensor

    def _preprocess_yolo(self, image: Image.Image) -> np.ndarray:
        """Resizes to 640x640 and normalizes to [0.0, 1.0] for YOLO."""
        img_resized = image.resize((640, 640), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)
        return tensor

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _run_yolo_detection(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes YOLO plant & weed detection, decodes output tensor (1, 6, 8400),
        and applies OpenCV NMS. Returns (plants, weeds).
        """
        orig_w, orig_h = image.size
        blob = self._preprocess_yolo(image)

        yolo_out = self.yolo_session.run(None, {self.yolo_input_name: blob})[0]
        preds = yolo_out[0].T  # (8400, 6)
        boxes = preds[:, :4]  # cx, cy, w, h in 640x640
        scores = preds[:, 4:]  # class 0: plant, class 1: weed

        x_factor = orig_w / 640.0
        y_factor = orig_h / 640.0

        detections_by_class = {0: [], 1: []}

        for cls_idx in [0, 1]:
            cls_scores = scores[:, cls_idx]
            mask = cls_scores >= conf_threshold
            filtered_boxes = boxes[mask]
            filtered_scores = cls_scores[mask]

            if len(filtered_boxes) == 0:
                continue

            nms_rects = []
            for b in filtered_boxes:
                cx, cy, bw, bh = b
                x1 = int((cx - bw / 2.0) * x_factor)
                y1 = int((cy - bh / 2.0) * y_factor)
                w_box = int(bw * x_factor)
                h_box = int(bh * y_factor)
                nms_rects.append([x1, y1, w_box, h_box])

            indices = cv2.dnn.NMSBoxes(
                nms_rects,
                [float(s) for s in filtered_scores],
                score_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )

            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, w_box, h_box = nms_rects[idx]
                    conf = float(filtered_scores[idx])
                    # Clamp coordinates to original image boundaries
                    x1_clamped = max(0, min(orig_w, x1))
                    y1_clamped = max(0, min(orig_h, y1))
                    x2_clamped = max(0, min(orig_w, x1 + w_box))
                    y2_clamped = max(0, min(orig_h, y1 + h_box))

                    detections_by_class[cls_idx].append({
                        "confidence": round(conf, 4),
                        "bounding_box": {
                            "x1": x1_clamped,
                            "y1": y1_clamped,
                            "x2": x2_clamped,
                            "y2": y2_clamped
                        }
                    })

        return detections_by_class[0], detections_by_class[1]

    def _draw_annotations(
        self,
        image: Image.Image,
        plants: List[Dict[str, Any]],
        weeds: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Draws localized bounding boxes onto a copy of the input image.
        Plants: Green (#2E7D32)
        Weeds: Orange/Red (#E53E3E)
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw Plants (Green)
        for p in plants:
            bb = p["bounding_box"]
            conf = p["confidence"]
            draw.rectangle([bb["x1"], bb["y1"], bb["x2"], bb["y2"]], outline="#2E7D32", width=3)
            label = f"Plant {conf:.2f}"
            label_w = len(label) * 7 + 6
            top_y = max(0, bb["y1"] - 15)
            draw.rectangle([bb["x1"], top_y, bb["x1"] + label_w, max(0, bb["y1"])], fill="#2E7D32")
            draw.text((bb["x1"] + 3, top_y + 1), label, fill="#FFFFFF", font=font)

        # Draw Weeds (Orange/Red)
        for w in weeds:
            bb = w["bounding_box"]
            conf = w["confidence"]
            draw.rectangle([bb["x1"], bb["y1"], bb["x2"], bb["y2"]], outline="#E53E3E", width=3)
            label = f"Weed {conf:.2f}"
            label_w = len(label) * 7 + 6
            top_y = max(0, bb["y1"] - 15)
            draw.rectangle([bb["x1"], top_y, bb["x1"] + label_w, max(0, bb["y1"])], fill="#E53E3E")
            draw.text((bb["x1"] + 3, top_y + 1), label, fill="#FFFFFF", font=font)

        return annotated

    def predict(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Executes the authoritative 4-stage pipeline:
        1. Crop Verification -> Halt if not soybean
        2. Disease Classification
        3. Potassium Deficiency Detection
        4. Plant / Weed Localization (YOLO)
        """
        image = self._load_image(image_input)

        # -------------------------------------------------------------
        # STEP 1: CROP VERIFICATION
        # -------------------------------------------------------------
        tensor_224 = self._preprocess_224(image)
        crop_logits = self.crop_session.run(None, {self.crop_input_name: tensor_224})[0]
        crop_probs = self._softmax(crop_logits)[0]
        # Class 0 = Soybean, Class 1 = Non_Soybean
        prob_soybean = float(crop_probs[0])
        is_soybean = bool(prob_soybean >= self.crop_threshold)

        if not is_soybean:
            return {
                "is_soybean": False,
                "soybean_confidence": round(prob_soybean, 4),
                "message": "Please upload a soybean image."
            }

        # -------------------------------------------------------------
        # STEP 2: DISEASE CLASSIFICATION (EfficientNetV2-M)
        # -------------------------------------------------------------
        disease_logits = self.disease_session.run(None, {self.disease_input_name: tensor_224})[0]
        disease_probs = self._softmax(disease_logits)[0]
        disease_best_idx = int(np.argmax(disease_probs))
        disease_label = self.disease_labels.get(disease_best_idx, f"Class_{disease_best_idx}")
        disease_conf = float(disease_probs[disease_best_idx])

        # -------------------------------------------------------------
        # STEP 3: POTASSIUM DEFICIENCY (EfficientNetV2-S)
        # -------------------------------------------------------------
        pot_logits = self.potassium_session.run(None, {self.potassium_input_name: tensor_224})[0]
        pot_probs = self._softmax(pot_logits)[0]
        # Class 0: Normal_Status, Class 1: Potassium_Deficiency
        prob_potassium_def = float(pot_probs[1])
        if prob_potassium_def >= self.potassium_threshold:
            potassium_status = "Potassium_Deficiency"
            potassium_conf = prob_potassium_def
        else:
            potassium_status = "Normal_Status"
            potassium_conf = float(pot_probs[0])

        # -------------------------------------------------------------
        # STEP 4: PLANT / WEED DETECTION (YOLO11-M)
        # -------------------------------------------------------------
        plant_detections, weed_detections = self._run_yolo_detection(image)

        # -------------------------------------------------------------
        # STEP 5: ANNOTATION VISUALIZATION
        # -------------------------------------------------------------
        annotated_img = self._draw_annotations(image, plant_detections, weed_detections)
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=88)
        visualization_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # -------------------------------------------------------------
        # STEP 6: ASSEMBLE CLEAN RESPONSE
        # -------------------------------------------------------------
        return {
            "is_soybean": True,
            "soybean_confidence": round(prob_soybean, 4),
            "disease": {
                "label": disease_label,
                "formatted_name": format_display_name(disease_label),
                "confidence": round(disease_conf, 4)
            },
            "potassium": {
                "status": potassium_status,
                "formatted_status": format_display_name(potassium_status),
                "confidence": round(potassium_conf, 4)
            },
            "plants": {
                "count": len(plant_detections),
                "detections": plant_detections
            },
            "weeds": {
                "count": len(weed_detections),
                "detections": weed_detections
            },
            "has_localized_detections": (len(plant_detections) > 0 or len(weed_detections) > 0),
            "visualization": visualization_url
        }
