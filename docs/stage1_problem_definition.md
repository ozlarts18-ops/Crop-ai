# Stage 1: Real-World Problem Definition - Soybean (Glycine max)

## 1. Problem Overview
Early, accurate diagnosis of soybean foliar diseases and pest attacks is vital for food security and minimizing yield losses.

## 2. Core Detection Targets
- **Plant Name:** Soybean
- **Scientific Name:** Glycine max
- **Health Status:** Healthy / Diseased
- **Disease Classification:** Identification across 11 normalized pathological classes
- **Confidence Scoring:** Calibrated probability distribution
- **Spatial Localization:** Object detection coordinates (where localization data exists)
- **Severity & Spreadness:** Disease area ratio / spread percentage (where segmentation data exists)

## 3. Normalized Disease Classes
- `0`: Healthy
- `1`: Bacterial_Blight
- `2`: Cercospora_Leaf_Blight
- `3`: Downy_Mildew
- `4`: Frogeye_Leaf_Spot
- `5`: Soybean_Rust
- `6`: Target_Spot
- `7`: Potassium_Deficiency
- `8`: Mosaic
- `9`: Septoria_Brown_Spot
- `10`: Sudden_Death_Syndrome
- `11`: Pest_Damage
