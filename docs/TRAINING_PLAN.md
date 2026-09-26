# Crop_AI — Master Training Plan & Architecture Specification
**Project:** Soybean Health & Crop AI Assistant  
**Target Deployment:** Simple Local/Web Application (Flask / Python backend)  
**Date:** September 20, 2026  
**Status:** Architecture Frozen — Training Blocked Pending Review  

---

## 1. Final Task Deconstruction & Model Allocation

Rather than forcing incompatible label spaces and modalities into a single brittle monolith, Crop_AI utilizes a modular, decoupled architecture where each model has a distinct, well-defined mathematical objective.

```
User Uploads Soybean Image
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Upload Validation & Image Quality Check        │
│ (Blur, brightness, resolution, aspect ratio check)      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Crop Species Verification                      │
│ (Model 4: Confirm authentic soybean vs. non-soybean)    │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│ Model 1: Disease      │   │ Model 2: Potassium    │
│ Classification        │   │ Deficiency Classifier │
│ (9 Canonical Classes) │   │ (K-Deficiency vs. OK) │
└───────────┬───────────┘   └───────────┬───────────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Growth Progression & Canopy Analysis           │
│ (Model 3: Temporal Canopy/Weed Density Estimation)      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Consolidated Crop Health Synthesis             │
│ (Overall Health Status, Actionable Agronomic Advisory)  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Model 1: Soybean Disease Classification

- **Target Task:** Whole-leaf multi-class image classification.
- **Source Datasets:**
  - Primary: `Multi-Class Soybean Leaf Disease Dataset Healthy a` (499 images)
  - Secondary: `MH-SoyaHealthVision (Leaf Subset)` (2,782 images)
- **Final Disease Classes (9 Classes):**
  1. `Bacterial_Blight`
  2. `Cercospora_Leaf_Blight`
  3. `Frogeye_Leaf_Spot`
  4. `Healthy`
  5. `Mosaic`
  6. `Pest_Damage`
  7. `Septoria_Brown_Spot`
  8. `Soybean_Rust`
  9. `Sudden_Death_Syndrome`
- **Recommended Architecture:** **ConvNeXt-Tiny** or **EfficientNet-B0**
  - Justification: Exceptional feature extraction for fine lesion textures, fast CPU/GPU inference suitable for real-time web rendering, strong transfer learning capabilities from ImageNet.
- **Input Resolution:** $224 \times 224 \times 3$ or $256 \times 256 \times 3$.

---

## 3. Model 2: Potassium Deficiency Classification

- **Target Task:** Single-class nutrient deficiency detection / binary classification.
- **Source Dataset:** `potassium_deficiency` (1,034 images) paired with healthy control leaf imagery from `Multi-Class`.
- **Classes:**
  - `Potassium_Deficiency`
  - `Normal_Potassium_Status` (Healthy leaf controls)
- **Unsupported Nutrient Deficiencies (Explicitly Documented):**
  - Nitrogen (N) Deficiency: **UNSUPPORTED (No training data)**
  - Phosphorus (P) Deficiency: **UNSUPPORTED (No training data)**
- **Recommended Architecture:** **MobileNetV3-Large** or **EfficientNet-B0**
  - Justification: Lightweight, robust gradient descent convergence on leaf-margin chlorosis/necrosis patterns without overfitting.

---

## 4. Model 3: Growth Progression & Canopy Monitoring

- **Target Task:** Chronological canopy progression and weed presence analysis.
- **Source Dataset:** `A soybean and weed image dataset collected using FarmBot` (1,300 images across Days 1–20).
- **Available Labels:** `Day 1` through `Day 20` daily observations; Roboflow bounding boxes for `plant` and `weed`.
- **Formal Agronomical Growth Stages:** **UNSUPPORTED**. The raw data contains zero vegetative (VE, V1, V2, V3) or reproductive (R1, R2, R3) stage annotations.
- **Recommended Model Design:**
  - Option A (Regression / Temporal Ordering): Image-to-Canopy-Progression score (0.0 to 1.0) indicating vegetative canopy coverage.
  - Option B (Detection): Lightweight YOLOv8n / YOLO11n object detector trained strictly on `plant` and `weed` to estimate weed infestation density.

---

## 5. Model 4: Optional Soybean Crop Verification

- **Target Task:** Binary crop verification (Soybean vs. Non-Soybean).
- **Source Dataset:** `SoyCotton` (640 images, 12,411 instances of `soy` vs `cotton`).
- **Classes:** `Soybean` vs `Non-Soybean / Cotton`.
- **Function in Web App:** Gating filter. When a user uploads a photo of a weed, tomato, or arbitrary object, the verifier intercepts it before disease inference, alerting the user to provide a valid soybean leaf.

---

## 6. Whether YOLO is Actually Needed

### Conclusion: YOLO is NOT NEEDED for Disease or Potassium Diagnosis.
- **Analysis:** Both `Multi-Class Soybean Leaf Disease` and `potassium_deficiency` consist of clear, centered leaf photographs. Whole-image classification provides higher diagnostic accuracy and faster web inference without requiring artificial bounding box fabrication.
- **Where YOLO CAN Be Justified:**
  - Only for **FarmBot Plant & Weed Localization** (where labels are natively bounding boxes for individual weeds in soil).
  - Or for **SoyCotton Field Foliage Localization**.
- **Web App Strategy:** The core web application will use CNN classifiers for Disease and Potassium, ensuring fast, deterministic, reliable inference.

---

## 7. Dataset Split Strategy & Class Imbalance Handling

- **Partition Ratios:** 70% Train, 15% Validation, 15% Test.
- **Zero-Leakage Enforcement:** Hash-grouped splitting via `data/metadata/` manifests to ensure duplicate files never cross splits.
- **Class Imbalance Mitigation:**
  - Multi-Class is already balanced (97–105/class).
  - When incorporating MH-SoyaHealthVision (which has 852 Rust vs. 169 Frogeye), apply **Focal Loss** ($\gamma = 2.0$) or class-weighted cross-entropy loss:
    $$w_c = \frac{N_{total}}{K \times N_c}$$

---

## 8. Data Augmentation Strategy

To ensure robust performance on real-world smartphone photos uploaded by farmers:
- **Spatial:** Random horizontal/vertical flips, random rotation ($\pm 25^\circ$), random perspective warping.
- **Color & Lighting:** ColorJitter (brightness $\pm 0.2$, contrast $\pm 0.2$, saturation $\pm 0.2$, hue $\pm 0.05$).
- **Environmental:** Random Gaussian blur (kernel size 3 to 5) to simulate camera autofocus errors.
- **Pristine Evaluation:** Validation and Test sets must receive **NO augmentations** (only standard resize and ImageNet normalization).

---

## 9. Evaluation Metrics
- **Multi-Class Disease:** Top-1 Accuracy, Macro-averaged Precision, Macro-averaged Recall, and Macro F1-Score:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Potassium Deficiency:** Sensitivity, Specificity, ROC-AUC.
- **Confusion Matrix:** Full $9 \times 9$ normalized confusion matrix to identify any confusion between Bacterial Blight and Cercospora Leaf Blight.

---

## 10. Expected Web Application Output Schema

When a user uploads an image, the website displays:
```json
{
  "image_quality": {
    "status": "PASS",
    "blur_score": 142.5,
    "brightness": "OPTIMAL"
  },
  "crop_verification": {
    "is_soybean": true,
    "confidence": 0.982
  },
  "disease_diagnostic": {
    "predicted_disease": "Soybean_Rust",
    "confidence": 0.941,
    "status": "UNHEALTHY"
  },
  "nutrient_diagnostic": {
    "potassium_deficiency_detected": false,
    "potassium_confidence": 0.965,
    "nitrogen_deficiency": "NOT ANALYZED (No model available)",
    "phosphorus_deficiency": "NOT ANALYZED (No model available)"
  },
  "growth_analysis": {
    "monitoring_support": "Temporal canopy monitoring available",
    "formal_growth_stage": "UNAVAILABLE"
  },
  "overall_health_assessment": {
    "health_status": "Diseased",
    "recommendation": "Apply recommended fungicide for Soybean Rust control."
  }
}
```

---

## 11. What CANNOT Currently Be Predicted (Missing Training Data)

To maintain absolute scientific and engineering integrity, the system will explicitly NOT display or claim the following capabilities:
1. **Affected-Area Percentage:** No dataset contains pixel-level lesion masks or validated percent-leaf-area-affected ground truth.
2. **Nitrogen (N) or Phosphorus (P) Deficiency:** No training images exist in the active datasets for N or P deficiency.
3. **Formal Agronomical Growth Stages (VE, V1, V2, R1, etc.):** The FarmBot dataset contains chronological days (Day 1–20), not agronomist-validated staging labels.
4. **Soil-Level Nutrient Concentrations:** Physical chemical values cannot be inferred from leaf RGB images without calibrated spectroscopy.
