import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
YOLO_DIR = os.path.join(PROCESSED_DIR, "yolo")
CNN_DIR = os.path.join(PROCESSED_DIR, "cnn")
SEG_DIR = os.path.join(PROCESSED_DIR, "segmentation")
METADATA_DIR = os.path.join(PROCESSED_DIR, "metadata")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Normalized Classes mapping as required
# 0 Healthy
# 1 Bacterial_Blight
# 2 Cercospora_Leaf_Blight
# 3 Downy_Mildew
# 4 Frogeye_Leaf_Spot
# 5 Soybean_Rust
# 6 Target_Spot
# 7 Potassium_Deficiency
# 8 Mosaic
# 9 Septoria_Brown_Spot
# 10 Sudden_Death_Syndrome
# 11 Pest_Damage

NORMALIZED_CLASSES = {
    0: "Healthy",
    1: "Bacterial_Blight",
    2: "Cercospora_Leaf_Blight",
    3: "Downy_Mildew",
    4: "Frogeye_Leaf_Spot",
    5: "Soybean_Rust",
    6: "Target_Spot",
    7: "Potassium_Deficiency",
    8: "Mosaic",
    9: "Septoria_Brown_Spot",
    10: "Sudden_Death_Syndrome",
    11: "Pest_Damage"
}

CLASS_NAME_TO_ID = {v: k for k, v in NORMALIZED_CLASSES.items()}

# Plant details
PLANT_NAME = "Soybean"
SCIENTIFIC_NAME = "Glycine max"

# Split Ratios
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
RANDOM_SEED = 42

# Ensure directories exist
for d in [DATA_DIR, PROCESSED_DIR, YOLO_DIR, CNN_DIR, SEG_DIR, METADATA_DIR, OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)
