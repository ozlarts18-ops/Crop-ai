# Website Architecture & AI Integration Analysis

**Project**: Crop_AI — Soybean Crop Health AI Scanner  
**Analysis Date**: September 20, 2026  
**Status**: Analysis Complete — Ready for Integration  

---

## 1. Executive Summary

Crop_AI is an existing, functional web application designed for soybean health diagnosis. The application consists of a lightweight Flask backend and a clean, responsive vanilla HTML/CSS/JavaScript frontend. Currently, the backend loads an older set of PyTorch models via `src/deployment/inference_service.py` (an EfficientNet-B0 CNN and three YOLO11m models for disease, nutrient, and leaf detection).

The goal of this project is to integrate the **authoritative trained ONNX models** into the **existing** application architecture, preserving the visual identity, simplicity, and user experience, while implementing the required 4-stage AI pipeline:
1. **Crop Verification** (EfficientNetV2-S ONNX) — Binary Soybean vs. Non-Soybean check with threshold gating (0.35).
2. **Disease Classification** (EfficientNetV2-M ONNX) — 9-class soybean disease classification.
3. **Potassium Deficiency Detection** (EfficientNetV2-S ONNX) — Binary Normal vs. Potassium Deficiency detection with threshold gating (0.45).
4. **Plant & Weed Detection** (YOLO11-M ONNX) — Object detection for plant and weed counts and localized bounding boxes.

---

## 2. Current Website Architecture

### 2.1 Technology Stack
- **Backend**: Python 3.12 / Flask (served via `python app.py` at `http://127.0.0.1:5000`)
- **Frontend**: 
  - Structure: Jinja2 HTML (`templates/index.html`)
  - Styling: Vanilla CSS (`static/style.css`)
  - Interactivity: Vanilla JavaScript (`static/app.js`)
- **ML / Runtime**:
  - `onnxruntime` 1.30.0 (fast CPU/GPU execution without PyTorch overhead)
  - `Pillow` (PIL) & `OpenCV` (cv2) for image preprocessing and bounding box rendering
  - `numpy` for tensor and probability manipulation

### 2.2 Frontend Structure
- **Entry File**: `templates/index.html`
- **Styling**: `static/style.css` (Clean card layout, responsive max-width 680px, neutral slate palette `#2b6cb0` primary)
- **Client Logic**: `static/app.js`
  - Drag-and-drop / file input selection (`#imageInput`, `#fileName`, `#btnChoose`)
  - Client-side image preview (`#imagePreview`)
  - Analyze action trigger (`#scanBtn`)
  - Status banner (`#statusMessage`)
  - Multi-section result display card (`#resultCard`):
    - Health status block
    - Disease detection block
    - Nutrient deficiency block
    - Leaf / weed detection block
    - Annotated image visualization with color legend
    - Model summary cards

### 2.3 Backend Structure
- **Entry File**: `app.py`
  - Routes:
    - `GET /`: Renders `templates/index.html`
    - `GET /health`: Health-check endpoint verifying model readiness
    - `POST /predict`: Handles multipart file upload (`image`), validates image format and size, runs inference, draws bounding boxes, and returns JSON.
- **Inference Layer**:
  - `src/deployment/inference_service.py`: Singleton model manager caching models at startup.
  - `src/deployment/model_loader.py`: Model loading routines.
  - `src/deployment/model_registry.py`: Catalog of model metadata.
  - `src/deployment/output_schema.py`: Canonical validation schema.

---

## 3. Existing User & Data Flow

```
1. User Selects Image (JPG/JPEG/PNG)
       │
       ▼
2. Client Preview Generated (FileReader DataURL) & "SCAN IMAGE" button enabled
       │
       ▼
3. User Clicks "SCAN IMAGE"
       │
       ▼
4. POST /predict (multipart/form-data: image)
       │
       ▼
5. Backend Validation (Presence, Extension, File Size <= 16MB)
       │
       ▼
6. Backend Inference Service
       │
       ▼
7. JSON Response Returned to Frontend
       │
       ▼
8. DOM Update (Populates Result Card, Displays Annotated Image, Hides Loading)
```

---

## 4. Trained AI Models to Integrate

| # | Task | Architecture | Artifact Path | Labels / Threshold Config | Key Behavior |
|---|---|---|---|---|---|
| 1 | **Crop Verification** | EfficientNetV2-S | `models/exports/crop_verification_efficientnetv2_s.onnx` | `models/labels/crop_verification_labels.json`<br>`models/labels/crop_verification_threshold.json` (th = 0.35) | If $P(\text{Soybean}) < 0.35$, halt pipeline and return: *"Please upload a soybean image."* |
| 2 | **Disease Classification** | EfficientNetV2-M | `models/exports/disease_efficientnetv2_m.onnx` | `models/labels/disease_labels.json` (9 classes) | Predicts primary disease and classification confidence. |
| 3 | **Potassium Deficiency** | EfficientNetV2-S | `models/exports/potassium_efficientnetv2_s.onnx` | `models/labels/potassium_labels.json`<br>`models/labels/potassium_threshold.json` (th = 0.45) | Exclusively detects *Normal* vs. *Potassium Deficiency*. |
| 4 | **Plant & Weed Detection** | YOLO11-M | `models/exports/plant_weed_yolo11_m.onnx` | `models/labels/plant_weed_labels.json` (`0: plant`, `1: weed`) | Localizes plants and weeds, counts detections, and provides bounding boxes. |

---

## 5. Integration Architecture (Smallest Possible Changes)

To adhere strictly to the project rules—preserving UI/UX, avoiding redesigns, and ensuring rock-solid stability:

1. **New High-Performance Inference Service** (`src/deployment/onnx_pipeline.py` or updating `inference_service.py`):
   - Initializes 4 ONNX Runtime sessions once at application startup.
   - Session reuse across all requests (zero repeated disk I/O or model loading).
   - Preprocessing with standardized PIL and NumPy array ops (224x224 normalized with ImageNet statistics for EfficientNet models, 640x640 normalized `[0, 1]` for YOLO).
   - Strict execution order:
     - Step 1: Crop Verifier. If not soybean, return `is_soybean: false` immediately.
     - Step 2: Concurrently or sequentially run Disease, Potassium, and YOLO detection.
     - Step 3: Compute bounding boxes with NMS for plant and weed instances.
     - Step 4: Draw clean, labeled bounding boxes onto the annotated image (Plants: Green, Weeds: Orange/Crimson).

2. **Backend Route (`app.py`)**:
   - Update `predict()` to consume the 4-model ONNX pipeline.
   - If image is not soybean: return `{"success": false, "error": "Please upload a soybean image."}` or `{"success": true, "is_soybean": false, "error": "Please upload a soybean image."}`.
   - Return formatted JSON compatible with the existing frontend expectations.

3. **Frontend Presentation (`templates/index.html` & `static/app.js`)**:
   - Keep existing cards, header, styles, and buttons.
   - Result card updates:
     - **Verification**: Soybean Verified (✓ Verified).
     - **Disease**: Formatted clean name (e.g., "Soybean Rust", "Bacterial Blight", or "Healthy") with confidence percentage.
     - **Potassium**: "Normal" or "Potassium Deficiency" with confidence percentage (no claim of nitrogen/phosphorus).
     - **Plants & Weeds**: Display plant count and weed count clearly.
     - **Annotated Image**: Show annotated bounding boxes for detected plants and weeds.
     - Suppress developer/engineering diagnostics from normal user view, keeping the experience clean and user-friendly.

---

## 6. File Modification Scope

### Files to Modify
- `app.py`: Hook up the ONNX inference service, update response payload formatting and error responses.
- `src/deployment/inference_service.py`: Refactor to load the 4 authoritative ONNX models via `onnxruntime` and implement the 4-step pipeline.
- `templates/index.html`: Minor text/label updates to align headings with Soybean Verification, Disease, Potassium Deficiency, and Plant/Weed counts (retaining the exact existing HTML structure).
- `static/app.js`: Adapt the JSON parser to map the new response format to the existing DOM elements, ensuring clean error alerts if the image is not a soybean.

### Files to NOT Modify
- `static/style.css`: Visual styling is already clean, professional, and responsive. No CSS redesign is needed.
- `models/exports/*.onnx`: Frozen trained models; must remain untouched.
- `models/labels/*.json`: Official label mappings and thresholds; must remain untouched.
- `src/train/*`, `experiments/*`, `data/*`: Historical training and evaluation pipelines.

---

## 7. Potential Blockers & Mitigations

1. **Ultralytics Dependency on `onnxruntime-gpu`**:
   - *Issue*: Loading YOLO `.onnx` via `ultralytics.YOLO()` triggers an automatic attempt to pip-install `onnxruntime-gpu`.
   - *Mitigation*: Run `plant_weed_yolo11_m.onnx` directly using `onnxruntime.InferenceSession` and perform straightforward NumPy box decoding and `cv2.dnn.NMSBoxes`. This is 100% self-contained, lightning-fast, and avoids any runtime pip installs.
2. **Non-Soybean Rejection User Experience**:
   - *Issue*: Non-soybean images must immediately stop the pipeline and notify the user.
   - *Mitigation*: Return a clear 400 response with `"Please upload a soybean image."` which the existing frontend renders directly into `#statusMessage` with error styling.
