# Model Card: Soybean Crop Verifier (EfficientNetV2-S)

## Model Details
- **Architecture:** EfficientNetV2-S
- **Pretrained Weights:** ImageNet-1k (`EfficientNet_V2_S_Weights.DEFAULT`)
- **Task:** Binary Crop Verification (Gating classifier before disease and nutrient inference)
- **Input Resolution:** $224 \times 224 \times 3$ (letterbox resize with bilinear interpolation, ImageNet normalized)
- **Output:** 2 logits / softmax probabilities:
  - Class 0: `Soybean`
  - Class 1: `Non_Soybean`
- **Framework:** PyTorch 2.6.0+cu124 / ONNX Opset 14

---

## Intended Use
- **Primary Function:** Acts as an automated input-validation gate for user-uploaded imagery in the Soybean Crop AI web assistant.
- **Workflow Position:**
  ```
  User Image
     ↓
  Crop Verification Model (EfficientNetV2-S)
     ↓
  Is Soybean? (P(Soybean) >= 0.35)
     ├─► YES ──► Disease Model (EfficientNetV2-M) & Potassium Model (EfficientNetV2-S)
     └─► NO  ──► Reject: Inform user that the image does not contain recognizable soybean foliage
  ```
- **Out-of-Scope Use:** This model is **NOT** a disease diagnostic tool, does **NOT** assess nutrient levels, and is **NOT** a general botanic plant identifier for arbitrary wild flora.

---

## Dataset & Training Data
- **Dataset:** SoyCotton
- **Total Images:** 1,154 verified images
- **Class Balance:** Perfectly balanced ($577$ Soybean, $577$ Non-Soybean / Cotton)
- **Splits (Deterministic & Hash-Isolated):**
  - **Train:** 806 images ($69.8\%$) — $403$ Soybean, $403$ Non-Soybean
  - **Validation:** 172 images ($14.9\%$) — $86$ Soybean, $86$ Non-Soybean
  - **Test:** 176 images ($15.3\%$) — $88$ Soybean, $88$ Non-Soybean
- **Leakage Prevention:** Grouped by exact MD5 hash; $0$ hash leakage across splits.

---

## Training Strategy & Hyperparameters
- **Optimization:** AdamW ($\text{weight decay} = 1 \times 10^{-4}$)
- **Stage 1 (Head Warmup):** 5 epochs, backbone frozen, $\text{LR} = 1 \times 10^{-3}$
- **Stage 2 (Full Fine-Tuning):** Full network unfrozen, $\text{LR} = 1 \times 10^{-4}$, CosineAnnealingLR ($\text{LR}_{\text{min}} = 1 \times 10^{-6}$), AMP enabled.
- **Checkpoint Selection:** Best epoch selected strictly by **Validation F1** (Epoch 19: $\text{Val F1} = 0.9942$).
- **Threshold Calibration:** Probability threshold tuned exclusively on validation data across $[0.30 - 0.70]$. Optimal threshold locked at **$0.35$** to ensure $100\%$ Soybean retention.

---

## Empirical Test Metrics (Untouched Test Partition)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Test Accuracy** | **99.43%** | 175 / 176 correct |
| **Test Precision (Soybean)** | **0.9888** | 88 true positives / 89 positive predictions |
| **Test Recall (Soybean)** | **1.0000** | **100.0%** (0 false negatives — 0 soybean images rejected) |
| **Test F1-Score** | **0.9944** | Harmonic mean of precision and recall |
| **ROC-AUC** | **1.0000** | Perfect separation on test partition |
| **PR-AUC** | **1.0000** | Area under precision-recall curve |
| **Non-Soybean Recall** | **98.86%** | 87 / 88 cotton images rejected |
| **Confusion Matrix** | `[[88, 0], [1, 87]]` | $\text{TP}=88, \text{FN}=0, \text{FP}=1, \text{TN}=87$ |

---

## Deployment Artifacts
- **Best Checkpoint:** `models/checkpoints/crop_verification_best.pt`
- **ONNX Model:** `models/exports/crop_verification_efficientnetv2_s.onnx`
- **Label Mapping:** `models/labels/crop_verification_labels.json`
- **Threshold File:** `models/labels/crop_verification_threshold.json`
- **Metadata Manifest:** `data/metadata/crop_verification_metadata.csv`

---

## Known Limitations
1. **Image Quality & Perspective:** Crop verification accuracy is dependent on clear focus. Severe motion blur or extreme lens distortion may degrade feature extraction.
2. **Extreme Viewpoints:** Imagery taken from unusual angles (e.g. extreme macro of stem hairs without foliage) may not match standard field foliage patterns.
3. **Heavy Occlusion:** Plants heavily occluded by soil, plastic mulch, or dense weed mats may reduce model confidence.
4. **Botanical Scope:** The negative training class consists of field cotton foliage. While the model distinguishes non-soybean dicot leaves effectively, it does **not** claim to identify every plant species in existence.
5. **No Pathology Diagnostic:** This model strictly verifies crop identity; it **never** determines whether a crop is healthy or diseased.
