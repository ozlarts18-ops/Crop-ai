# Crop_AI — Final Comprehensive Dataset Audit Report
**Project:** Soybean Health & Crop AI Assistant  
**Date:** September 20, 2026  
**Deployment Target:** Simple Web Application  
**Status:** Audit Complete — No Models Trained — Raw Datasets Preserved  

---

## Executive Summary
This document provides an exhaustive, empirical audit of all datasets residing within `raw/` for the Crop_AI Soybean Health system. The target deployment is a responsive, web-based AI assistant. In accordance with project decisions:
- Original raw datasets are 100% preserved without modification or deletion.
- YOLO is not arbitrarily forced; image classification is prioritized for whole-leaf disease and nutrient assessment, while object detection/segmentation is reserved strictly for tasks with native localization requirements (e.g., FarmBot weed/plant detection, SoyCotton crop verification).
- No models have been trained or converted; this audit freezes and validates dataset integrity and class boundaries prior to training.

---

## Phase 1: High-Level Inventory of Active Datasets

| Dataset Identifier | Physical Path | Total Files | Total Images | Primary Image Formats | Typical Dimensions | Color Mode | Primary Annotation Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Class Soybean Leaf Disease** | `raw/Multi-Class Soybean Leaf Disease Dataset Healthy a` | 499 | 499 | `.jpg` (100%) | $5472 \times 3648$ | RGB (3-ch) | Directory structure (Classification) |
| **MH-SoyaHealthVision (Leaf)** | `raw/MH-SoyaHealthVision An Indian UAV.../Soyabean_Leaf_Image_Dataset` | 2,782 | 2,782 | `.jpg` (100%) | $3000 \times 4000$ to $4000 \times 3000$ | RGB (3-ch) | Zip-folder structure (Classification) |
| **MH-SoyaHealthVision (UAV)** | `raw/MH-SoyaHealthVision An Indian UAV.../Soyabean_UAV-Based_Image_Dataset` | 2,842 | 2,842 | `.jpg` (100%) | $3840 \times 2160$ to $4000 \times 3000$ | RGB (3-ch) | Zip-folder structure (Classification) |
| **potassium_deficiency** | `raw/potassium_deficiency/potassium_deficiency` | 1,035 | 1,034 | `.jpg` (100%) | $4000 \times 3000$ | RGB (3-ch) | Directory structure (Single-class) |
| **FarmBot Soybean & Weed** | `raw/A soybean and weed image dataset collected using F/.../Soyaben-Weed Dataset.zip` | 2,100 | 1,300 | `.jpg` (100%) | $1600 \times 1200$ | RGB (3-ch) | YOLO `.txt` bounding boxes (plant, weed) |
| **SoyCotton** | `raw/SoyCotton/SoyCotton` | 641 | 640 | `.jpeg` (51.6%), `.jpg` (48.4%) | $1600 \times 1200$ to $1200 \times 1200$ | RGB (3-ch) | COCO JSON (`coco.json`) instances |
| **SoyNet (Raw Media)** | `raw/SoyNet Indian Soybean Image dataset with quality i/SoyNet/Raw_SoyNet_Data` | 3,655 | 3,655 | `.jpg` (100%) | $3024 \times 4032$ (Variable) | RGB (3-ch) | Directory structure (Disease vs Healthy) |
| **SoyNet (Preprocessed)** | `raw/SoyNet Indian Soybean Image dataset with quality i/SoyNet/Preprocessing_SoyNet_Data` | 14,292 | 14,292 | `.jpg` (100%) | $256 \times 256$ | RGB / Grayscale | Synthetic/Resized Variants |

---

## Phase 2: In-Depth Dataset-by-Dataset Analysis

### A. MH-SoyaHealthVision
- **Exact Path:** `raw/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment` (also aliased via directory junction `raw/MH-SoyaHealthVision`).
- **Archive Structure:** Contains 10 large `.zip` files totaling ~10.1 GB.
- **Image Breakdown:**
  - **Leaf-Level Image Subset:** 2,782 images
    - `Soybean Rust` (`Soyabean_Rust.zip`): 852 images
    - `Soybean Mosaic` (`Soyabean_Mosaic.zip`): 707 images
    - `Caterpillar and Semilooper Pest Attack` (`Caterpillar and Semilooper Pest Attack.zip`): 582 images
    - `Septoria Brown Spot` (`Soyabean_Spectoria_Brown_Spot.zip`): 268 images
    - `Healthy Soybean` (`Healthy_Soyabean.zip`): 204 images
    - `Frogeye Leaf Spot` (`Soyabean_Frog_Leaf_Eye.zip`): 169 images
  - **UAV-Based Aerial Subset:** 2,842 images
    - `Soybean Rust`: 1,000 images
    - `Caterpillar & Semilooper Pest Attack`: 790 images
    - `Soybean Mosaic`: 772 images
    - `Healthy Soybean`: 280 images
- **Annotation Reality vs. Literature:** Although published research describes COCO polygon masks, the local raw dataset contains **0 annotation files**; all images are organized into category folders inside `.zip` archives.
- **Image Perspective & Quality:** Handheld smartphone close-up shots under real Indian field lighting and background soil/canopy.
- **Critical Leakage / Quality Finding:** 20 duplicate images exist identically in both `Caterpillar and Semilooper Pest Attack` and `Soyabean_Mosaic` zips (e.g. `20240914_112955.jpg`).
- **Recommendation:**
  - **Leaf Subset:** Use as a secondary expansion dataset to provide Frogeye, Mosaic, Septoria, and Pest Damage to the primary classifier (after unzipping and deduplication).
  - **UAV Subset:** **MUST BE EXCLUDED** from the leaf diagnostic pipeline. Drone canopy shots from 10–30m altitude have completely different visual features and will severely degrade a leaf-level classifier.

---

### B. SoyNet Indian Soybean Image Dataset
- **Exact Path:** `raw/SoyNet Indian Soybean Image dataset with quality i/SoyNet`
- **Total Images:** 17,947 images across raw and preprocessed directories.
- **Taxonomy / Labels:**
  - `Raw_SoyNet_Data/Camera Clicks/Disease_Pic`: 2,762 images
  - `Raw_SoyNet_Data/Camera Clicks/Healthy_pic`: 445 images
  - `Raw_SoyNet_Data/Mobile pic`: 448 images
  - `Preprocessing_SoyNet_Data`: 14,292 images consisting of $256 \times 256$ resized crops, grayscale conversions, and augmented duplicates (`aug_*.jpg`).
- **Core Limitation:** SoyNet labels are strictly **binary** (`Disease` vs `Healthy`). It does **NOT** distinguish between Rust, Blight, Frogeye, or Mosaic.
- **Duplicates & Redundancy:** 123 internal duplicate image pairs found in the raw folder. Thousands of preprocessed images are simple algorithmic rescales of the raw set.
- **Recommendation:**
  - **Do NOT use SoyNet for multi-class disease training.**
  - Use `Raw_SoyNet_Data` as an external evaluation pool for:
    1. Upload image-quality verification (blur/illumination check).
    2. General preliminary health gating (Healthy vs. Unhealthy).

---

### C. Multi-Class Soybean Leaf Disease Dataset
- **Exact Path:** `raw/Multi-Class Soybean Leaf Disease Dataset Healthy a/Soyabean leaf desease dataset`
- **Total Images:** 499 high-resolution images.
- **Class Breakdown:**
  - `Bacterial Blight`: 99 images (19.84%)
  - `Cercospora Leaf Blight`: 99 images (19.84%)
  - `Healthy`: 97 images (19.44%)
  - `Rust`: 99 images (19.84%)
  - `Sudden Death Syndrome`: 105 images (21.04%)
- **Image Characteristics:** Uniform $5472 \times 3648$ high-resolution macro photography with exceptional clarity, natural lighting, and consistent leaf centering.
- **Balance & Integrity:** Perfectly balanced (imbalance ratio $1.08:1$). 30 internal identical image pairs detected (`BB(59)` == `BB(60)`).
- **Recommendation:** **PRIMARY DISEASE CLASSIFICATION DATASET CANDIDATE.** It forms the gold-standard foundation for the soybean disease classifier.

---

### D. potassium_deficiency Dataset
- **Exact Path:** `raw/potassium_deficiency/potassium_deficiency`
- **Total Images:** 1,034 images (plus 1 hidden `.DS_Store`).
- **Class Breakdown:** 1,034 images of **Potassium Deficiency** exclusively.
- **Nutrient Scope Verification:**
  - **Potassium (K) Deficiency:** AVAILABLE (1,034 images).
  - **Nitrogen (N) Deficiency:** NOT AVAILABLE (0 images).
  - **Phosphorus (P) Deficiency:** NOT AVAILABLE (0 images).
  - **Calcium / Magnesium:** NOT AVAILABLE (0 images).
- **Image Characteristics:** $4000 \times 3000$ high-resolution field photographs showing marginal leaf chlorosis and necrosis characteristic of potassium deficiency.
- **Duplicates:** Zero duplicate images internally ($1,034 / 1,034$ unique hashes). Zero overlap with other repository datasets.
- **Recommendation:** Dedicated dataset for **MODEL 2: Potassium Deficiency Classification**. The web UI must state clearly that only Potassium deficiency is analyzed; do NOT pretend to predict N or P deficiency.

---

### E. FarmBot / Soybean and Weed Dataset
- **Exact Path:** `raw/A soybean and weed image dataset collected using F/A soybean and weed image dataset collected using F/Soyaben-Weed Dataset.zip`
- **Archive Size:** 779.24 MB (2,100 valid non-Mac files, 1,300 unique `.jpg` images).
- **Image Breakdown:**
  - **Annotated Images:** 641 images across 20 consecutive days (`Day 1` through `Day 20`).
  - **Unannotated Images:** 659 images.
- **Annotations:** Roboflow YOLOv8/v11 format with bounding box coordinates for two classes:
  `names: ['plant', 'weed']`
- **Growth Information Analysis:**
  - Captures the chronological canopy development of soybean seedlings in a controlled FarmBot outdoor automated bed over a 20-day temporal sequence.
  - **Crucial Finding:** Contains **NO formal agronomical growth-stage labels** (such as VE, V1, V2, V3, R1, R2).
- **Recommendation:** Formally classify as **temporal growth monitoring data**. The web application will display chronological canopy progression / weed density monitoring, NOT fake VE/R1 stage classifications.

---

### F. SoyCotton Dataset
- **Exact Path:** `raw/SoyCotton/SoyCotton`
- **Total Images:** 640 images ($1600 \times 1200$ to $1200 \times 1200$).
- **Annotations:** Standard COCO JSON (`coco.json`) with 12,411 annotated polygon instances:
  - `soy` (category id 1): 11,288 instances (canopy foliage)
  - `cotton` (category id 2): 1,123 instances (neighboring/intercropped foliage)
- **Health / Disease Content:** None. All plants are healthy vegetative field foliage.
- **Recommendation:** Formally assign as **MODEL 4: Soybean Crop Verifier / Plant Segmentation**. Used to verify that an uploaded photograph actually contains soybean foliage and reject non-soybean images before disease inference.

---

## Phase 3: Task-to-Dataset Suitability Matrix

| Dataset | Disease Classification | Potassium Deficiency | Growth Monitoring | Crop Verification | Object Localization |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Class Leaf Disease** | **EXCELLENT (Primary)** | No | No | No | No |
| **MH-SoyaHealthVision (Leaf)** | **GOOD (Secondary)** | No | No | No | No |
| **MH-SoyaHealthVision (UAV)** | Unsuitable (Aerial) | No | No | No | Optional (Aerial) |
| **potassium_deficiency** | No | **EXCELLENT (K only)** | No | No | No |
| **FarmBot Soybean & Weed** | No | No | **EXCELLENT (Days 1-20)**| No | Yes (`plant`, `weed`) |
| **SoyCotton** | No | No | No | **EXCELLENT** | Yes (`soy`, `cotton`) |
| **SoyNet (Raw)** | Screening Only | No | No | No | No |

---

## Phase 4: Quality & Integrity Findings
1. **Zero Corrupted Images:** Every inspected JPG/JPEG across all datasets successfully opened, decoded, and validated via PIL and OpenCV.
2. **Resolution Discrepancies:**
   - Multi-Class: $5472 \times 3648$ (high-end DSLR).
   - Potassium: $4000 \times 3000$ (smartphone macro).
   - FarmBot: $1600 \times 1200$ (automated gantry camera).
   - SoyCotton: $1600 \times 1200$ (handheld field).
   - SoyNet: $256 \times 256$ to $3024 \times 4032$.
   - **Recommendation:** Standardize classifier training input resolution to $224 \times 224$ or $256 \times 256$ with aspect-ratio preserving letterbox resizing.
3. **Internal Duplication:**
   - Multi-Class contains 30 duplicate pairs.
   - FarmBot contains 59 duplicate pairs across Day 5 and Day 12 valid folders.
   - MH-SoyaHealthVision contains 20 conflicting duplicates between Pest and Mosaic.
   - All duplicates are recorded in [data/metadata/](file:///c:/Users/oswal/Music/Crop_AI/data/metadata) manifests to enforce strict partition isolation.
