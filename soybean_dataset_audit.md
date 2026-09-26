# Soybean Dataset Cleanup & Comprehensive Audit Report
**Project:** Crop Health AI / Smart Farming Assistant  
**Architecture Mode:** Classification-First Pipeline (Target: ESP32-S3 / Edge Deployment)  
**Date:** September 20, 2026  
**Status:** Audit Complete — Datasets Cleaned & Prepared  

---

## 1. Datasets Found
An initial recursive scan of the `raw/` directory identified 7 folder entries (representing 6 distinct physical datasets/directories):

| Folder / Entry Name | Type | Physical Entity | Description |
| :--- | :--- | :--- | :--- |
| `MH-SoyaHealthVision` | Directory Junction / Symlink | Link to `MH-SoyaHealthVision An Indian UAV...` | Convenient shortcut symlink |
| `MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment` | Directory | 10 Zip Archives (~10.1 GB) | Maharashtra UAV and Leaf image collection |
| `Multi-Class Soybean Leaf Disease Dataset Healthy a` | Directory | 499 High-Res Images | 5-class balanced leaf disease dataset |
| `Nutrient Deficiency Obj.v1i.yolov11` | Directory | 2,133 Files (51.60 MB) | Roboflow YOLOv11 nutrient detection dataset |
| `Soybean Crop Disease.v10-version_1.yolov11` | Directory | 5,155 Files (926.66 MB) | Roboflow YOLOv11 disease detection dataset |
| `SoyCotton` | Directory | 641 Files (COCO annotations + images) | Soybean vs. Cotton instance dataset |
| `SoyNet Indian Soybean Image dataset with quality i` | Directory | 17,947 Images | Indian soybean quality/health image dataset |

---

## 2. YOLO Datasets Removed
Per explicit architectural decision to **discontinue YOLO object detection** in favor of an ESP32-S3 deployable classification-first architecture, the two YOLO-specific dataset directories have been permanently removed from `raw/`:

1. **`raw/Nutrient Deficiency Obj.v1i.yolov11`**
   - **Files removed:** 2,133 files (images + `.txt` YOLO bounding box annotations + YAML configs)
   - **Storage freed:** 51.60 MB
   - **Status:** REMOVED

2. **`raw/Soybean Crop Disease.v10-version_1.yolov11`**
   - **Files removed:** 5,155 files (images + `.txt` YOLO bounding box annotations + YAML configs)
   - **Storage freed:** 926.66 MB
   - **Status:** REMOVED

**Total Storage Recovered:** 978.26 MB (7,288 files).

---

## 3. YOLO-Related Project References Removed & Updated
1. **`configs/deployment.yaml`**: Updated service specification to `version: 2.0.0-classification`, `architecture_mode: classification_first`, and `target_hardware: ESP32-S3 / Edge`. YOLO detector models marked as decommissioned.
2. **Historical Training & Audit Scripts**: Audited and isolated historical scripts (`src/train_new_yolo_test_models.py`, `src/evaluate_and_report_new_models.py`, `src/audit_new_datasets.py`) so they no longer execute in active deployment workflows.
3. **Scratch Files**: Removed temporary YOLO inspection and search scripts.

---

## 4. Remaining Datasets
The active soybean dataset collection now consists strictly of non-YOLO image collections:

1. **`Multi-Class Soybean Leaf Disease Dataset Healthy a`**
2. **`MH-SoyaHealthVision`** (and target `MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment`)
3. **`SoyCotton`**
4. **`SoyNet Indian Soybean Image dataset with quality i`**

---

## 5. Image Counts
| Dataset Name | Total Images | Image Formats | Resolution Range | Total Size |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Class Soybean Leaf Disease** | 499 | `.jpg` (100%) | $5472 \times 3648$ (Uniform) | ~1.4 GB |
| **MH-SoyaHealthVision (Leaf)** | 2,782 | `.jpg` (100%) | $3000 \times 4000$ to $4000 \times 3000$ | ~5.8 GB |
| **MH-SoyaHealthVision (UAV)** | 2,842 | `.jpg` (100%) | $3840 \times 2160$ to $4000 \times 3000$ | ~4.3 GB |
| **SoyCotton** | 640 | `.jpeg` (51.6%), `.jpg` (48.4%) | $1200 \times 1200$ to $1600 \times 1600$ | ~280 MB |
| **SoyNet (Raw Images)** | 3,655 | `.jpg` (100%) | $256 \times 256$ to $3120 \times 4160$ | ~850 MB |
| **SoyNet (Preprocessed/Augmented)** | 14,292 | `.jpg` (100%) | $256 \times 256$ | ~420 MB |
| **Active Total Pool** | **24,710** | — | — | **~13.0 GB** |

---

## 6. Class Counts
| Dataset | Number of Classes | Annotation Type | Primary Task |
| :--- | :--- | :--- | :--- |
| **Multi-Class Soybean Leaf Disease** | 5 classes | Folder-based classification | Multi-Class Leaf Disease Classification |
| **MH-SoyaHealthVision (Leaf)** | 6 classes | Folder-based classification (in zips) | Multi-Class Leaf Disease Classification |
| **MH-SoyaHealthVision (UAV)** | 4 classes | Folder-based classification (in zips) | Aerial Field-Level Stress Classification |
| **SoyCotton** | 2 classes (`soy`, `cotton`) | COCO instance segmentation | Crop Species Verification |
| **SoyNet** | 2 classes (`Disease`, `Healthy`) | Folder-based classification | Quality Check & Binary Health Screening |

---

## 7. Class Distributions
### A. Multi-Class Soybean Leaf Disease Dataset (499 Images)
- **Bacterial Blight**: 99 images (19.84%)
- **Cercospora Leaf Blight**: 99 images (19.84%)
- **Healthy**: 97 images (19.44%)
- **Rust**: 99 images (19.84%)
- **Sudden Death Syndrome**: 105 images (21.04%)
- **Distribution Assessment**: Exceptionally balanced (min: 97, max: 105). Imbalance ratio: $1.08:1$.

### B. MH-SoyaHealthVision — Leaf Subset (2,782 Images)
- **Soybean Rust**: 852 images (30.63%)
- **Soybean Mosaic**: 707 images (25.41%)
- **Caterpillar and Semilooper Pest Attack**: 582 images (20.92%)
- **Septoria Brown Spot** (`Soyabean_Spectoria_Brown_Spot`): 268 images (9.63%)
- **Healthy Soybean**: 204 images (7.33%)
- **Frogeye Leaf Spot** (`Soyabean_Frog_Leaf_Eye`): 169 images (6.07%)
- **Distribution Assessment**: Moderately imbalanced (min: 169, max: 852). Imbalance ratio: $5.04:1$.

### C. MH-SoyaHealthVision — UAV Subset (2,842 Images)
- **Soybean Rust**: 1,000 images (35.19%)
- **Soybean Semilooper and Caterpillar Pest Attack**: 790 images (27.80%)
- **Soybean Mosaic**: 772 images (27.16%)
- **Healthy Soybean**: 280 images (9.85%)
- **Distribution Assessment**: Moderate imbalance (min: 280, max: 1,000). Imbalance ratio: $3.57:1$.

### D. SoyCotton (640 Images, 12,411 Instances)
- **Soybean Instances (`soy`)**: 11,288 annotations (90.95%)
- **Cotton Instances (`cotton`)**: 1,123 annotations (9.05%)
- **Distribution Assessment**: Heavily dominated by soybean canopy plants.

### E. SoyNet — Raw Capture Data (3,655 Images)
- **Camera Clicks — Disease**: 2,762 images (75.57%)
- **Camera Clicks — Healthy**: 445 images (12.18%)
- **Mobile Clicks — Mixed/Disease**: 448 images (12.26%)
- **Distribution Assessment**: Heavily imbalanced toward diseased plants (~85% disease, ~15% healthy).

---

## 8. Disease Classes (Taxonomy Mapping)
Across all remaining datasets, the following 9 fine-grained disease/damage classes and 1 binary disease class are represented:

| Canonical Disease Name | Datasets Containing Class | Total Available Leaf Images |
| :--- | :--- | :--- |
| **Bacterial Blight** | `Multi-Class` | 99 |
| **Cercospora Leaf Blight** | `Multi-Class` | 99 |
| **Frogeye Leaf Spot** | `MH-SoyaHealthVision (Leaf)` | 169 |
| **Mosaic** | `MH-SoyaHealthVision (Leaf)` | 707 |
| **Pest Damage / Caterpillar Attack** | `MH-SoyaHealthVision (Leaf)` | 582 |
| **Rust** | `Multi-Class`, `MH-SoyaHealthVision (Leaf)` | 951 (99 + 852) |
| **Septoria Brown Spot** | `MH-SoyaHealthVision (Leaf)` | 268 |
| **Sudden Death Syndrome** | `Multi-Class` | 105 |
| **Generic Disease (Unspecified)** | `SoyNet` | 2,762 raw (+ 8,082 preprocessed) |

---

## 9. Nutrient-Deficiency Classes
> [!IMPORTANT]
> **Zero Nutrient-Deficiency Data in Remaining Datasets:**  
> The remaining datasets (`Multi-Class`, `MH-SoyaHealthVision`, `SoyCotton`, and `SoyNet`) contain **NO nutrient-deficiency classes** (Nitrogen, Phosphorus, Potassium, Calcium, Magnesium, etc.).
>
> **Architectural Recommendation:**  
> Model B (Soybean Nutrient Deficiency Classification) cannot be trained from the remaining raw datasets. Nutrient-deficiency classification must remain a separate model pipeline to be populated once dedicated, non-YOLO classification imagery is acquired. Do NOT mix nutrient deficiency into the disease classifier.

---

## 10. Healthy Classes
| Dataset | Folder / Class Label | Image Count | Visual Characteristics |
| :--- | :--- | :--- | :--- |
| `Multi-Class` | `Healthy` | 97 | Clean, vibrant green, single-leaf macro shots |
| `MH-SoyaHealthVision (Leaf)` | `Healthy_Soyabean` | 204 | Real field conditions, variable lighting, healthy trifoliates |
| `MH-SoyaHealthVision (UAV)` | `Healthy_Soyabean` | 280 | Aerial drone overhead shots of healthy crop plots |
| `SoyNet (Raw)` | `Healthy_pic` | 445 | Smartphone/camera field clicks |
| `SoyNet (Preprocessed)` | `Grayscale_Healthy_data` | 1,374 | Grayscale $256 \times 256$ processed variants |

Total unique healthy leaf photographs available: **746 raw leaf images** (plus 280 UAV shots).

---

## 11. UAV / Field / Leaf Image Categories
- **Leaf-Level Imagery (Macro / Handheld)**:
  - `Multi-Class Soybean Leaf Disease`: 100% individual leaf images (macro close-up).
  - `MH-SoyaHealthVision (Leaf)`: 100% handheld smartphone close-up photos taken in Maharashtra fields.
  - `SoyNet`: Close-up leaves and small canopies.
- **UAV / Aerial Imagery**:
  - `MH-SoyaHealthVision (UAV)`: 100% drone captures from DJI aerial platforms. Broad canopy perspective, unsuitable for leaf-level CNN classification.
- **Canopy / Crop Verification Imagery**:
  - `SoyCotton`: Canopy rows containing mixed soybean and cotton plants.

---

## 12. Annotation Types
- **Classification Labels**: Present in `Multi-Class`, `MH-SoyaHealthVision`, and `SoyNet` (encoded via directory structures).
- **Object Detection / Localization Bounding Boxes**: None in remaining disease datasets (YOLO datasets removed).
- **Instance Segmentation**: Present only in `SoyCotton` (`coco.json` with polygon coordinates for `soy` and `cotton` plant instances).
- **Pixel-Level Disease Lesion Masks**: **NONE**. Neither `Multi-Class` nor `MH-SoyaHealthVision` contains pixel-level lesion masks or polygon annotations for disease spots.

---

## 13. Duplicate Detection Findings
Audit performed across all datasets using MD5 checksums:

### A. Internal Exact Duplicates (Within Same Dataset)
1. **`Multi-Class`**: **30 sets of duplicate images (31 duplicate files)**.
   - Examples: `BB(59).jpg` == `BB(60).jpg`, `BB(88).jpg` == `BB(90).jpg`, `CLB(11).jpg` == `CLB(92).jpg`.
2. **`SoyNet (Raw)`**: **123 sets of duplicate images (141 duplicate files)**.
   - Numerous `aug_*.jpg` files in the raw folder are bit-for-bit identical to unaugmented files (e.g. `aug_18385.jpg` == `soy (78).jpg`).
3. **`MH-SoyaHealthVision`**: **20 duplicate images across conflicting categories**.
   - Identical images exist in both `Caterpillar and Semilooper Pest Attack` and `Soyabean_Mosaic` (e.g., `20240914_112955.jpg`, `20240914_112959.jpg`, `20240914_113024.jpg`).

### B. Cross-Dataset Duplicates
- **Zero cross-dataset exact duplicates**: No bit-for-bit identical files are shared across `Multi-Class`, `SoyCotton`, `SoyNet`, and `MH-SoyaHealthVision`.

---

## 14. Data Leakage Protection
To guarantee zero data leakage between training, validation, and test splits:
1. **Hash-Grouped Splitting**: All identical duplicate pairs (such as `BB(59)` and `BB(60)`) must be grouped by MD5 hash before partitioning, ensuring duplicates never cross into validation or test sets.
2. **Exclusion of Preprocessed SoyNet Copies**: When using SoyNet, only use the `Raw_SoyNet_Data` subset; avoid `Preprocessing_SoyNet_Data` to eliminate duplicate/synthetic leakage.
3. **Cross-Class Conflict Pruning**: The 20 identical images in `MH-SoyaHealthVision` shared between `Pest Attack` and `Mosaic` must be resolved/removed to prevent ambiguous ground truth.

---

## 15. Dataset Overlaps
1. **Physical Directory Symlink**: `MH-SoyaHealthVision` is a direct NTFS junction to `MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment`. They must not be treated as two independent datasets.
2. **Taxonomic Overlaps**:
   - `Rust` is represented in both `Multi-Class` (99 images) and `MH-SoyaHealthVision Leaf` (852 images).
   - `Healthy` is represented in `Multi-Class` (97 images), `MH-SoyaHealthVision Leaf` (204 images), and `SoyNet` (445 images).

---

## 16. Recommended Dataset Roles
| Dataset | Recommended Role | Justification |
| :--- | :--- | :--- |
| **`Multi-Class Soybean Leaf Disease`** | **Primary Disease Dataset Candidate** | Pristine image quality, balanced classes (97–105 images/class), verified clean disease presentation. |
| **`MH-SoyaHealthVision (Leaf)`** | **Secondary Disease Expansion Dataset** | Adds 4 critical field diseases (Frogeye, Mosaic, Septoria Brown Spot, Caterpillar Pest Attack) after unzipping and deduplication. |
| **`SoyCotton`** | **Soybean Crop Verifier** | Dedicated verification stage to confirm leaf/plant is soybean and reject non-soybean foliage. |
| **`SoyNet (Raw)`** | **Quality Assurance & Binary Health Pre-screener** | High-volume real-world camera/phone clicks for image quality verification (blur, illumination) and preliminary healthy/unhealthy gating. |
| **`MH-SoyaHealthVision (UAV)`** | **Separate Aerial Model (Excluded from Leaf Pipeline)** | Aerial drone canopy perspective cannot be combined with close-up leaf classification. |

---

## 17. Classes Requiring Manual Review
1. **`Soyabean_Spectoria_Brown_Spot`**: Typo in original folder name for *Septoria Brown Spot* (*Septoria glycines*). Rename to canonical `Septoria_Brown_Spot`.
2. **`Soyabean_Frog_Leaf_Eye`**: Inverted phrasing for *Frogeye Leaf Spot* (*Cercospora sojina*). Rename to canonical `Frogeye_Leaf_Spot`.
3. **`Caterpillar and Semilooper Pest Attack`**: Mechanical feeding damage caused by chewing insects rather than a microbial pathogen. Should be standardized as `Pest_Damage`.
4. **Conflicting Pest vs. Mosaic Duplicates**: The 20 images shared between Pest Attack and Mosaic in `MH-SoyaHealthVision` must be visually inspected and assigned to the correct single class.

---

## 18. What Should NOT Be Merged
- **DO NOT merge UAV imagery with handheld leaf photos**: Drone canopy images have vastly different ground sampling distances and perspective.
- **DO NOT merge SoyCotton with disease datasets**: SoyCotton contains no disease annotations; merging it into a disease classifier would introduce unlabelled healthy foliage as false negatives.
- **DO NOT merge SoyNet's generic "Disease" class into fine-grained disease classes**: SoyNet does not specify the pathogen; grouping it would blur the decision boundaries of Rust, Frogeye, Blight, etc.
- **DO NOT merge without hash-based deduplication**: Internal duplicates must not be allowed to leak into separate evaluation partitions.

---

## 19. What Can Potentially Be Merged
Once individual datasets are extracted and deduplicated, **`Multi-Class`** and **`MH-SoyaHealthVision (Leaf)`** can be harmonized into a canonical **9-Class Soybean Health Classification Taxonomy**:

1. `Bacterial_Blight` (`Multi-Class`)
2. `Cercospora_Leaf_Blight` (`Multi-Class`)
3. `Frogeye_Leaf_Spot` (`MH-SoyaHealthVision Leaf`)
4. `Healthy` (`Multi-Class` + `MH-SoyaHealthVision Leaf`)
5. `Mosaic` (`MH-SoyaHealthVision Leaf`)
6. `Pest_Damage` (`MH-SoyaHealthVision Leaf`)
7. `Septoria_Brown_Spot` (`MH-SoyaHealthVision Leaf`)
8. `Soybean_Rust` (`Multi-Class` + `MH-SoyaHealthVision Leaf`)
9. `Sudden_Death_Syndrome` (`Multi-Class`)

*Note: This exactly aligns with the 9-class CNN taxonomy previously validated in Stage 8!*

---

## 20. Recommended Train / Validation / Test Strategy
- **Partition Ratios**: 70% Train / 15% Validation / 15% Test.
- **Grouping Policy**: Hash-grouped stratification (GroupKFold on image MD5).
- **Test Set Integrity**: The test set must consist solely of original, unaugmented field photographs.
- **Target Hardware Architecture**:
  - Backbone: MobileNetV3-Small or EfficientNet-B0.
  - Input resolution: $224 \times 224 \times 3$.
  - Quantization target: INT8 TFLite for ESP32-S3 (with ESP-NN acceleration).

---

## 21. What is Still Missing for Affected-Area Estimation
- **No Ground Truth Lesion Masks**: None of the remaining datasets contain pixel-level binary or semantic segmentation masks demarcating disease lesions on leaves.
- **No Percent Affected Area Ground Truth**: There are no agronomist-validated severity percentages (e.g. 15% leaf area affected) in the metadata.
- **Technical Conclusion**:
  - Affected-area percentage **cannot be trained using supervised learning** on the current dataset collection.
  - Estimating affected area must be formally deferred as a **later capability**.
  - Any future implementation will require either acquiring semantic segmentation mask datasets (e.g. Mobile-Unet / Fast-SCNN) or developing a validated classical color-segmentation heuristic (e.g. Otsu/HSV leaf-vs-lesion thresholding) benchmarked against physical leaf scans.

---

## Canonical Audit Summary

```
YOLO STATUS:
REMOVED (Nutrient Deficiency Obj.v1i.yolov11 and Soybean Crop Disease.v10-version_1.yolov11 permanently deleted from raw/)

REMAINING DATASETS:
1. Multi-Class Soybean Leaf Disease Dataset Healthy a (499 images)
2. MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset (5,624 images in 10 zips; 2,782 leaf images)
3. SoyCotton (640 images with COCO annotations)
4. SoyNet Indian Soybean Image dataset with quality i (17,947 images; 3,655 raw)

PRIMARY DISEASE DATASET CANDIDATE:
Multi-Class Soybean Leaf Disease Dataset Healthy a

SECONDARY/VALIDATION DATASETS:
- MH-SoyaHealthVision (Leaf Subset) for 4-class disease expansion
- SoyNet (Raw Subset) for image quality check and binary health screening

NUTRIENT DEFICIENCY DATA:
NONE in remaining datasets (must be sourced as a separate classification dataset for Model B)

DATASETS REQUIRING REVIEW:
- MH-SoyaHealthVision (20 conflicting duplicate images between Pest Attack and Mosaic)
- SoyCotton (pure crop verification, no disease labels)

AFFECTED AREA:
UNSUPPORTED on current datasets (no pixel-level lesion masks or severity ground truth; deferred as a later capability)

NEXT STEP:
Freeze the soybean class list and dataset composition only after reviewing this audit.
```
