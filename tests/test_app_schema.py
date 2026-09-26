"""
Stage 10 — Application & Schema Unit Tests
Verifies that:
1. Uploaded image formats (JPG, PNG) can be accepted and parsed into PIL Images.
2. Prediction results strictly comply with the Stage 9 canonical JSON schema.
3. The JSON validator accepts valid predictions.
4. Invalid spreadness / severity fields are rejected.
5. Missing required fields are rejected.

NOTE: Controlled application-level tests only. Does NOT touch quarantined test sets.
"""

import sys
import io
import json
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.deployment.output_schema import (
    validate_crop_ai_output,
    CropAIPredictionOutput,
    DiseaseDetection,
    NutrientDeficiency,
    DetectionItem,
    BoundingBox
)


class TestCropAIAppSchema(unittest.TestCase):

    def setUp(self):
        # Create dummy synthetic images in-memory (JPG and PNG)
        self.dummy_rgb_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        
        # In-memory JPG bytes
        jpg_buf = io.BytesIO()
        Image.fromarray(self.dummy_rgb_array).save(jpg_buf, format="JPEG")
        self.jpg_bytes = jpg_buf.getvalue()
        
        # In-memory PNG bytes
        png_buf = io.BytesIO()
        Image.fromarray(self.dummy_rgb_array).save(png_buf, format="PNG")
        self.png_bytes = png_buf.getvalue()

    def test_01_uploaded_image_can_be_accepted(self):
        """Test that uploaded image bytes (JPEG and PNG) are successfully opened and read."""
        # Test JPEG reading
        img_jpg = Image.open(io.BytesIO(self.jpg_bytes))
        self.assertEqual(img_jpg.format, "JPEG")
        self.assertEqual(img_jpg.size, (224, 224))

        # Test PNG reading
        img_png = Image.open(io.BytesIO(self.png_bytes))
        self.assertEqual(img_png.format, "PNG")
        self.assertEqual(img_png.size, (224, 224))

        # Check conversion to RGB
        rgb_converted = img_png.convert("RGB")
        self.assertEqual(rgb_converted.mode, "RGB")

    def test_02_prediction_result_follows_stage9_schema(self):
        """Test that a canonical CropAIPredictionOutput matches the schema structure."""
        out = CropAIPredictionOutput(
            crop="Soybean",
            health_status="Soybean_Rust",
            confidence=0.9321,
            disease_detection=DiseaseDetection(
                detected=True,
                disease_type="Target Leaf Spot",
                detections=[
                    DetectionItem(
                        confidence=0.915,
                        bounding_box=BoundingBox(x1=50, y1=40, x2=180, y2=160)
                    )
                ]
            ),
            nutrient_deficiency=NutrientDeficiency(
                detected=False,
                type=None,
                detections=[]
            )
        )
        d = out.to_dict()
        is_valid, msg = validate_crop_ai_output(d)
        self.assertTrue(is_valid, f"Validation failed: {msg}")
        self.assertEqual(d["crop"], "Soybean")
        self.assertIn("health_status", d)
        self.assertIn("confidence", d)
        self.assertIn("disease_detection", d)
        self.assertIn("nutrient_deficiency", d)
        self.assertTrue(isinstance(d["confidence"], float))
        self.assertTrue(d["disease_detection"]["detected"])
        self.assertFalse(d["nutrient_deficiency"]["detected"])

    def test_03_json_validator_accepts_valid_predictions(self):
        """Test that healthy, single-detection, and multi-detection valid predictions pass."""
        # Case A: Healthy
        healthy_payload = {
            "crop": "Soybean",
            "health_status": "Healthy",
            "confidence": 0.982,
            "disease_detection": {
                "detected": False,
                "disease_type": None,
                "detections": []
            },
            "nutrient_deficiency": {
                "detected": False,
                "type": None,
                "detections": []
            }
        }
        valid_a, msg_a = validate_crop_ai_output(healthy_payload)
        self.assertTrue(valid_a, f"Healthy payload failed: {msg_a}")

        # Case B: Multi-detection Disease
        multi_payload = {
            "crop": "Soybean",
            "health_status": "Septoria_Brown_Spot",
            "confidence": 0.884,
            "disease_detection": {
                "detected": True,
                "disease_type": "Charcol rot",
                "detections": [
                    {
                        "confidence": 0.85,
                        "bounding_box": {"x1": 10, "y1": 20, "x2": 80, "y2": 90}
                    },
                    {
                        "confidence": 0.76,
                        "bounding_box": {"x1": 110, "y1": 120, "x2": 190, "y2": 210}
                    }
                ]
            },
            "nutrient_deficiency": {
                "detected": False,
                "type": None,
                "detections": []
            }
        }
        valid_b, msg_b = validate_crop_ai_output(multi_payload)
        self.assertTrue(valid_b, f"Multi-detection payload failed: {msg_b}")

        # Case C: Nutrient Deficiency
        nutrient_payload = {
            "crop": "Soybean",
            "health_status": "Healthy",
            "confidence": 0.82,
            "disease_detection": {
                "detected": False,
                "disease_type": None,
                "detections": []
            },
            "nutrient_deficiency": {
                "detected": True,
                "type": "Potassium_deficiency",
                "detections": [
                    {
                        "confidence": 0.81,
                        "bounding_box": {"x1": 40, "y1": 50, "x2": 160, "y2": 170}
                    }
                ]
            }
        }
        valid_c, msg_c = validate_crop_ai_output(nutrient_payload)
        self.assertTrue(valid_c, f"Nutrient payload failed: {msg_c}")

    def test_04_invalid_spreadness_fields_are_rejected(self):
        """Test that any forbidden spreadness or severity fields are strictly rejected."""
        forbidden_keys = [
            "spreadness",
            "spread_percentage",
            "disease_area",
            "severity",
            "affected_area"
        ]

        for fk in forbidden_keys:
            bad_payload = {
                "crop": "Soybean",
                "health_status": "Soybean_Rust",
                "confidence": 0.90,
                fk: 0.45,  # Forbidden field
                "disease_detection": {
                    "detected": True,
                    "disease_type": "Target Leaf Spot",
                    "detections": [
                        {
                            "confidence": 0.89,
                            "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
                        }
                    ]
                },
                "nutrient_deficiency": {
                    "detected": False,
                    "type": None,
                    "detections": []
                }
            }
            is_valid, msg = validate_crop_ai_output(bad_payload)
            self.assertFalse(is_valid, f"Expected rejection for forbidden key '{fk}', but it passed!")
            self.assertIn("Forbidden field", msg)

    def test_05_missing_required_fields_are_rejected(self):
        """Test that missing required root and nested fields are rejected."""
        base_valid = {
            "crop": "Soybean",
            "health_status": "Healthy",
            "confidence": 0.95,
            "disease_detection": {
                "detected": False,
                "disease_type": None,
                "detections": []
            },
            "nutrient_deficiency": {
                "detected": False,
                "type": None,
                "detections": []
            }
        }

        # Root missing fields
        for rk in ["crop", "health_status", "confidence", "disease_detection", "nutrient_deficiency"]:
            corrupt = base_valid.copy()
            del corrupt[rk]
            is_valid, msg = validate_crop_ai_output(corrupt)
            self.assertFalse(is_valid, f"Expected rejection when missing '{rk}', but it passed!")
            self.assertIn(f"Missing required root field '{rk}'", msg)

        # Inverted bounding box rejection
        invalid_bb_payload = {
            "crop": "Soybean",
            "health_status": "Soybean_Rust",
            "confidence": 0.91,
            "disease_detection": {
                "detected": True,
                "disease_type": "Target Leaf Spot",
                "detections": [
                    {
                        "confidence": 0.85,
                        "bounding_box": {"x1": 200, "y1": 100, "x2": 150, "y2": 250}  # x2 < x1
                    }
                ]
            },
            "nutrient_deficiency": {
                "detected": False,
                "type": None,
                "detections": []
            }
        }
        is_valid_bb, msg_bb = validate_crop_ai_output(invalid_bb_payload)
        self.assertFalse(is_valid_bb, "Expected rejection for inverted x coordinates, but it passed!")


if __name__ == "__main__":
    unittest.main()
