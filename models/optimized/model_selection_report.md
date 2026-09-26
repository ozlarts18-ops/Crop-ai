# Crop_AI Stage 8: Model Selection Report

## Best Selected Model: `efficientnet_b0` (EXP_01_B0_Baseline)

- **Validation Accuracy:** 0.9088
- **Validation Macro F1:** 0.8812
- **Validation Weighted F1:** 0.9082
- **Model Size:** 16.14 MB (4.02M params)
- **Checkpoint Path:** `C:\Users\oswal\Music\Crop_AI\models\optimized\best_model.pt`

### Architectural Benchmarks

| Experiment | Architecture | Val Accuracy | Val Macro F1 | Val Weighted F1 | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXP_01_B0_Baseline | efficientnet_b0 | 0.9088 | 0.8812 | 0.9082 | 16.14 MB (4.02M params) |
| EXP_02_ResNet18 | resnet18 | 0.8934 | 0.8521 | 0.8943 | 42.65 MB (11.18M params) |
| EXP_03_MobileNetV3 | mobilenet_v3_large | 0.9088 | 0.8683 | 0.9092 | 16.07 MB (4.21M params) |
| EXP_04_EfficientNetB1 | efficientnet_b1 | 0.9079 | 0.8748 | 0.9091 | 24.89 MB (6.52M params) |

### Selection Rationale
The efficientnet_b0 achieved the highest validation macro F1 (0.8812) and balanced classification across both dominant and minority soybean disease classes. Its parameter efficiency (16.14 MB (4.02M params)) provides optimal inference latency.
