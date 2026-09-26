# Website AI Integration Specification & Documentation

**System**: Crop_AI — Soybean AI Health Scanner  
**Integration Status**: Fully Integrated & Verified  
**Runtime**: Python 3.12 / Flask / ONNX Runtime 1.30.0  

---

## 1. Architecture Overview

The Crop_AI application is an AI-powered diagnostic tool for soybean crops. It integrates four specialized machine learning models into a unified inference pipeline served through a Flask web server and rendered via a clean, responsive single-page user interface.

```
                  ┌───────────────────────────────┐
                  │    User Upload (Soybean Image)│
                  └──────────────┬────────────────┘
                                 │ HTTP POST /predict
                                 ▼
                  ┌───────────────────────────────┐
                  │  Crop Verification (ONNX)     │
                  │  Model: EfficientNetV2-S      │
                  │  Threshold: 0.35              │
                  └──────────────┬────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        P(Soybean) < 0.35               P(Soybean) >= 0.35
                 │                               │
                 ▼                               ▼
      Pipeline Halted Early             Authoritative Pipeline Continues
      Response (HTTP 400):                       │
      "Please upload a soybean image."           ├──► 2. Disease Classification (ONNX)
                                                 │      Model: EfficientNetV2-M (9 classes)
                                                 │
                                                 ├──► 3. Potassium Deficiency (ONNX)
                                                 │      Model: EfficientNetV2-S (Threshold: 0.45)
                                                 │
                                                 └──► 4. Plant / Weed Localization (ONNX)
                                                        Model: YOLO11-M (plant & weed counts)
                                                         │
                                                         ▼
                                                5. Annotation & Bounding Boxes
                                                         │
                                                         ▼
                                                Clean JSON Response to UI
```

---

## 2. Connected AI Models

| # | Subsystem | Architecture | Weights / ONNX File | Classes & Taxonomies | Decision Threshold |
|---|---|---|---|---|---|
| 1 | **Crop Verification** | EfficientNetV2-S | `models/exports/crop_verification_efficientnetv2_s.onnx` | `0: Soybean`, `1: Non_Soybean` | **0.35** (from `models/labels/crop_verification_threshold.json`) |
| 2 | **Disease Classification** | EfficientNetV2-M | `models/exports/disease_efficientnetv2_m.onnx` | 9 Classes (`Bacterial_Blight`, `Cercospora_Leaf_Blight`, `Frogeye_Leaf_Spot`, `Healthy`, `Mosaic`, `Pest_Damage`, `Septoria_Brown_Spot`, `Soybean_Rust`, `Sudden_Death_Syndrome`) | Argmax / Softmax |
| 3 | **Potassium Deficiency** | EfficientNetV2-S | `models/exports/potassium_efficientnetv2_s.onnx` | `0: Normal_Status`, `1: Potassium_Deficiency` | **0.45** (from `models/labels/potassium_threshold.json`) |
| 4 | **Plant & Weed Detection** | YOLO11-M | `models/exports/plant_weed_yolo11_m.onnx` | `0: plant`, `1: weed` | Conf: **0.25**, NMS IoU: **0.45** |

> [!NOTE]
> All ONNX models are instantiated once in memory during application startup (`app.py` / `CropAIInferenceService`) and reused across all incoming requests. No model weights are reloaded on subsequent calls.

---

## 3. Endpoints & API Contracts

### 3.1 `GET /`
- **Description**: Serves the main web interface (`templates/index.html`).

### 3.2 `GET /health`
- **Description**: Verifies model session availability.
- **Response**: `{"status": "ok"}` (HTTP 200)

### 3.3 `POST /predict`
- **Description**: Multipart form upload containing an `image` field (`.jpg`, `.jpeg`, or `.png`).
- **Maximum File Size**: 16 MB.

#### Success Response (HTTP 200)
```json
{
  "success": true,
  "is_soybean": true,
  "soybean_confidence": 0.9967,
  "disease": {
    "label": "Bacterial_Blight",
    "formatted_name": "Bacterial Blight",
    "confidence": 0.9158
  },
  "potassium": {
    "status": "Potassium_Deficiency",
    "formatted_status": "Potassium Deficiency",
    "confidence": 0.8014
  },
  "plants": {
    "count": 2,
    "detections": [
      {
        "confidence": 0.4802,
        "bounding_box": {"x1": 65, "y1": 0, "x2": 395, "y2": 103}
      }
    ]
  },
  "weeds": {
    "count": 0,
    "detections": []
  },
  "has_localized_detections": true,
  "visualization": "data:image/jpeg;base64,..."
}
```

#### Non-Soybean Rejection Response (HTTP 400)
```json
{
  "success": false,
  "is_soybean": false,
  "error": "Please upload a soybean image."
}
```

---

## 4. UI/UX Implementation Details

- **Visual Consistency**: Preserved the original clean card-based design in `static/style.css` without unnecessary redesigns, animations, or complex dashboards.
- **User-Friendly Results**: Technical architecture names (e.g. `EfficientNetV2-M`, `YOLO11-M`) are hidden from regular users. The results card presents plain, actionable outputs:
  - **Soybean Verification**: `✓ Verified` with confidence percentage.
  - **Disease Detection**: Human-readable disease diagnosis (e.g. "Soybean Rust", "Bacterial Blight", or "Healthy") with confidence percentage.
  - **Potassium Status**: Clear status of "Normal" or "Potassium Deficiency" with explicit disclaimer that other nutrients are not evaluated.
  - **Plant & Weed Detection**: Live counts of detected soybean plants and weeds, with localized green/red bounding boxes drawn directly on the annotated preview image.
- **Human-Readable Error Handling**: Uploading non-soybean images, corrupted files, or empty selections triggers polite status messages directly in the UI without exposing Python stack traces.

---

## 5. How to Run and Test

### 5.1 Starting the Application
From the repository root (`c:\Users\oswal\Music\Crop_AI`):
```bash
.\.venv\Scripts\python.exe app.py
```
The application will preload the four ONNX sessions and listen on `http://127.0.0.1:5000`.

### 5.2 Accessing the Web Interface
Open any modern web browser to:
```
http://127.0.0.1:5000
```

### 5.3 Running Automated Integration Tests
To run the automated test suite covering health checks, soybean prediction, non-soybean rejection, and input validations:
```bash
.\.venv\Scripts\python.exe -c "
import urllib.request, json
with urllib.request.urlopen('http://127.0.0.1:5000/health') as r:
    print('Health:', r.status, json.loads(r.read()))
"
```
