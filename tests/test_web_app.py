"""
Crop_AI — Web Application Unit & Integration Tests (Stage 10 Updated)
Tests Flask routes, image processing, multi-model outputs (CNN, Disease, Nutrient, Leaf),
on-page display structure, error handling, and confirms NO JSON download route exists.
Does NOT use quarantined test sets.
"""

import sys
import io
import json
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from src.deployment.output_schema import validate_crop_ai_output


class TestCropAIWebApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Prepare synthetic in-memory JPG
        img_jpg = Image.new("RGB", (300, 300), color=(40, 160, 40))
        d_jpg = ImageDraw.Draw(img_jpg)
        d_jpg.ellipse([50, 50, 250, 250], fill=(60, 190, 60))
        d_jpg.line([150, 50, 150, 250], fill=(20, 100, 20), width=2)
        jpg_buf = io.BytesIO()
        img_jpg.save(jpg_buf, format="JPEG")
        cls.jpg_bytes = jpg_buf.getvalue()

        # Prepare synthetic in-memory PNG
        img_png = Image.new("RGB", (300, 300), color=(40, 160, 40))
        d_png = ImageDraw.Draw(img_png)
        d_png.ellipse([50, 50, 250, 250], fill=(60, 190, 60))
        png_buf = io.BytesIO()
        img_png.save(png_buf, format="PNG")
        cls.png_bytes = png_buf.getvalue()

    def test_01_get_index_works(self):
        """Test that GET / returns status 200 and renders the HTML template."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Crop_AI", html)
        self.assertIn("Soybean AI Health Scanner", html)
        self.assertIn("SCAN IMAGE", html)
        self.assertIn("Choose Soybean Image", html)
        # Ensure NO download-json link exists
        self.assertNotIn("/download-json", html)
        self.assertNotIn("DOWNLOAD JSON", html)

    def test_02_health_route_works(self):
        """Test that GET /health returns status 200 and status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")

    def test_03_upload_accepts_jpg_and_predict_returns_full_model_results(self):
        """Test that POST /predict accepts JPG and produces all required model outputs."""
        data = {
            "image": (io.BytesIO(self.jpg_bytes), "test_leaf.jpg")
        }
        response = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json.get("success"))

        # Check Health Status (CNN)
        self.assertIn("health_status", res_json)
        hs = res_json["health_status"]
        self.assertIn("status", hs)
        self.assertIn("confidence", hs)
        self.assertIsInstance(hs["confidence"], float)

        # Check Disease Detection (YOLO)
        self.assertIn("disease_detection", res_json)
        dd = res_json["disease_detection"]
        self.assertIn("detected", dd)
        self.assertIn("count", dd)
        self.assertIn("detections", dd)

        # Check Nutrient Deficiency (YOLO)
        self.assertIn("nutrient_deficiency", res_json)
        nd = res_json["nutrient_deficiency"]
        self.assertIn("detected", nd)
        self.assertIn("count", nd)
        self.assertIn("detections", nd)

        # Check Soybean Leaf Detection (YOLO)
        self.assertIn("leaf_detection", res_json)
        ld = res_json["leaf_detection"]
        self.assertIn("detected", ld)
        self.assertIn("count", ld)
        self.assertIn("detections", ld)

        # Check Model Results Summary Card
        self.assertIn("model_results", res_json)
        mr = res_json["model_results"]
        self.assertEqual(mr["cnn"]["model_name"], "EfficientNet-B0")
        self.assertEqual(mr["disease_yolo"]["model_name"], "YOLO11m")
        self.assertEqual(mr["nutrient_yolo"]["model_name"], "YOLO11m")
        self.assertEqual(mr["leaf_yolo"]["model_name"], "YOLO11m")

        # Check Visualization
        self.assertIn("visualization", res_json)
        self.assertTrue(res_json["visualization"].startswith("data:image/jpeg;base64,"))

    def test_04_upload_accepts_png_and_predict_works(self):
        """Test that POST /predict accepts PNG and produces valid results."""
        data = {
            "image": (io.BytesIO(self.png_bytes), "test_leaf.png")
        }
        response = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("health_status", res_json)

    def test_05_invalid_file_rejected_with_exact_messages(self):
        """Test that missing image or invalid file formats return exact expected error messages."""
        # A. Missing image field completely
        resp_no_field = self.client.post("/predict", data={})
        self.assertEqual(resp_no_field.status_code, 400)
        self.assertEqual(resp_no_field.get_json().get("error"), "Please select a soybean image.")

        # B. Empty filename
        resp_empty_name = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_empty_name.status_code, 400)
        self.assertEqual(resp_empty_name.get_json().get("error"), "Please select a soybean image.")

        # C. Non-image file extension (.txt)
        bad_text_file = {
            "image": (io.BytesIO(b"Not an image"), "notes.txt")
        }
        resp_bad_ext = self.client.post(
            "/predict",
            data=bad_text_file,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_bad_ext.status_code, 400)
        self.assertEqual(resp_bad_ext.get_json().get("error"), "Please upload a JPG, JPEG or PNG image.")

        # D. Corrupted image bytes with .jpg extension
        corrupted_file = {
            "image": (io.BytesIO(b"CORRUPTED_BYTES_NOT_IMAGE"), "corrupt.jpg")
        }
        resp_corrupt = self.client.post(
            "/predict",
            data=corrupted_file,
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_corrupt.status_code, 400)
        self.assertEqual(resp_corrupt.get_json().get("error"), "Please upload a JPG, JPEG or PNG image.")

    def test_06_zero_spreadness_or_severity_fields(self):
        """Test that neither spreadness nor severity are generated in the response."""
        data = {
            "image": (io.BytesIO(self.jpg_bytes), "sample.jpg")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        res = response.get_json()

        for forbidden in ["spreadness", "spread_percentage", "disease_area", "severity", "affected_area"]:
            self.assertNotIn(forbidden, res)
            self.assertNotIn(forbidden, res.get("disease_detection", {}))
            self.assertNotIn(forbidden, res.get("nutrient_deficiency", {}))
            self.assertNotIn(forbidden, res.get("leaf_detection", {}))

    def test_07_no_download_json_route(self):
        """Test that GET /download-json does NOT exist (must return 404)."""
        response = self.client.get("/download-json")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
