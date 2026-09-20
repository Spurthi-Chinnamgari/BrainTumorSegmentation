import os

# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "brats2023",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

# ======================================================
# Patient
# ======================================================

PATIENT_ID = "BraTS-GLI-00000-000"

PATIENT_DIR = os.path.join(
    DATASET_ROOT,
    PATIENT_ID
)

# ======================================================
# MRI
# ======================================================

MODALITIES = [
    "t1n",
    "t1c",
    "t2w",
    "t2f"
]

PATCH_SIZE = (64, 64, 64)

SEGMENTATION = "seg"

SPLIT_SEED = 42
DEFAULT_PATIENT_PERCENTAGE = 100

# ======================================================
# Visualization
# ======================================================

FIGURE_SIZE = (16,5)
DEFAULT_SLICE = None

# ======================================================
# Processed Data
# ======================================================

PROCESSED_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

# ======================================================
# Dataset Split
# ======================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ======================================================
# Dataset Split
# ======================================================

SPLIT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits"
)