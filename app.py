"""
Crop_AI — Simple Flask Web Application
Soybean AI Health Scanner

Runs with:
    python app.py
Serves at:
    http://127.0.0.1:5000
"""

import sys
import io
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, jsonify
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.deployment.inference_service import CropAIInferenceService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB maximum upload

# Global model service cache
INFERENCE_SERVICE: Optional[CropAIInferenceService] = None


def get_inference_service() -> CropAIInferenceService:
    """
    Initializes or retrieves the cached multi-model inference service.
    Authoritative ONNX models are loaded once and reused across all requests.
    """
    global INFERENCE_SERVICE
    if INFERENCE_SERVICE is None:
        try:
            print("Initializing authoritative Crop_AI ONNX models...")
            INFERENCE_SERVICE = CropAIInferenceService()
            print("Crop_AI models loaded and cached successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise RuntimeError(f"AI model could not be loaded: {e}")
    return INFERENCE_SERVICE


@app.route("/", methods=["GET"])
def index():
    """Renders the main Crop_AI web page."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check route: returns ok only if models can be loaded."""
    try:
        get_inference_service()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an uploaded image, executes the authoritative 4-stage AI pipeline:
    1. Crop Verification (EfficientNetV2-S ONNX)
    2. Disease Classification (EfficientNetV2-M ONNX)
    3. Potassium Deficiency (EfficientNetV2-S ONNX)
    4. Plant / Weed Localization (YOLO11-M ONNX)
    """
    # 1. Validate file presence
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Please select a soybean image."}), 400

    file = request.files["image"]
    if not file or file.filename.strip() == "":
        return jsonify({"success": False, "error": "Please select a soybean image."}), 400

    # 2. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return jsonify({"success": False, "error": "Please upload a JPG, JPEG or PNG image."}), 400

    # 3. Read and decode image
    try:
        file_bytes = file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return jsonify({"success": False, "error": "Unable to read this image. Please upload a valid JPG, JPEG or PNG image."}), 400

    # 4. Access cached inference service
    try:
        service = get_inference_service()
    except Exception:
        return jsonify({"success": False, "error": "AI service is currently unavailable. Please try again."}), 500

    # 5. Execute Authoritative ONNX Pipeline
    try:
        result = service.predict(image)
    except Exception:
        return jsonify({"success": False, "error": "Unable to analyze this image. Please try another clear soybean image."}), 500

    # 6. Check Crop Verification Gate
    if not result.get("is_soybean", False):
        return jsonify({
            "success": False,
            "is_soybean": False,
            "error": "Please upload a soybean image."
        }), 400

    # 7. Return Successful Analysis
    response_data = {
        "success": True,
        **result
    }
    return jsonify(response_data), 200


if __name__ == "__main__":
    # Pre-warm model cache on startup
    try:
        get_inference_service()
    except Exception as e:
        print(f"Warning: Models could not be pre-warmed: {e}")

    print("\n" + "=" * 60)
    print("DESOYA Flask Web Application is running on http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
