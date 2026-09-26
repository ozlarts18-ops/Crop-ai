"""
End-to-End Application Pipeline Test for Stage 10
Tests inference pipeline on test images and verifies full prediction cycle.
"""

import sys
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.deployment.inference_service import CropAIInferenceService
from src.deployment.output_schema import validate_crop_ai_output
from app import draw_bounding_boxes

def run_end_to_end_test():
    print("Initializing CropAIInferenceService...")
    service = CropAIInferenceService()
    print("Service initialized successfully.")

    # Create a synthetic green leaf image
    img = Image.new("RGB", (640, 640), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    # Add leaf details
    draw.ellipse([100, 100, 540, 540], fill=(46, 170, 46), outline=(20, 100, 20), width=4)
    draw.line([320, 100, 320, 540], fill=(20, 100, 20), width=3)
    # Add a mock spot
    draw.ellipse([250, 250, 310, 310], fill=(139, 69, 19))

    # Test 1: JPG
    jpg_path = ROOT_DIR / "tests" / "sample_soybean.jpg"
    img.save(jpg_path, "JPEG")
    print(f"\nSaved test JPG to {jpg_path}")

    print("Running prediction on test JPG...")
    payload_jpg = service.predict(jpg_path)
    print("Prediction output (JPG):", json.dumps(payload_jpg, indent=2))
    
    is_valid, msg = validate_crop_ai_output(payload_jpg)
    assert is_valid, f"JPG prediction schema validation failed: {msg}"
    print(f"[OK] JPG prediction strictly complies with Stage 9 canonical schema.")

    # Test 2: PNG
    png_path = ROOT_DIR / "tests" / "sample_soybean.png"
    img.save(png_path, "PNG")
    print(f"\nSaved test PNG to {png_path}")

    print("Running prediction on test PNG...")
    payload_png = service.predict(png_path)
    print("Prediction output (PNG):", json.dumps(payload_png, indent=2))

    is_valid, msg = validate_crop_ai_output(payload_png)
    assert is_valid, f"PNG prediction schema validation failed: {msg}"
    print(f"[OK] PNG prediction strictly complies with Stage 9 canonical schema.")

    # Test 3: Visualization Function
    annotated = draw_bounding_boxes(
        img,
        payload_jpg.get("disease_detection", {}),
        payload_jpg.get("nutrient_deficiency", {})
    )
    assert isinstance(annotated, Image.Image)
    print(f"[OK] Visualization drawing function verified.")

    print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_end_to_end_test()
