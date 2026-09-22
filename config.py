"""
Central configuration for the JalSaafi oil-spill detection project.
Keeping paths/hyperparameters in one place means every script (inspect,
train, predict) stays in sync if the dataset location or settings change.
"""

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------

# Project root = the folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent

# The raw dataset lives outside the project folder (it's shared with other
# parts of the JalSaafi system). We point at it rather than copying it, so
# the project doesn't duplicate 1000s of images. Everything that DERIVES
# from the data (splits, models, plots, reports) is written inside
# jalsaafi-detection/, so the project stays self-contained.
#
# Dataset identity (confirmed via kaggle/metadata/dataset_metadata.xml):
#   "CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features for
#   machine learning (Deep Learning)", David Blondeau-Patissier, CSIRO,
#   2022-12-15. DOI: 10.25919/4v55-dn16. Licence: CC BY-SA 4.0 (attribution
#   + share-alike required — cite this in the slide deck).
#   Full collection: N=5,630 chips, Class 0 (no oil)=3,725 (66%),
#   Class 1 (oil)=1,905 (34%). Chips are 400x400, grayscale (1-band SAR),
#   saved as JPEG.
# Override with the JALSAAFI_RAW_DATA_DIR environment variable; defaults to
# <project>/kaggle/data. The dataset is NOT redistributed in this repo.
RAW_DATA_DIR = Path(os.environ.get("JALSAAFI_RAW_DATA_DIR", PROJECT_ROOT / "kaggle" / "data"))

# Expected class subfolders inside RAW_DATA_DIR (there is also a "sample"
# subfolder — a small preview duplicate set, NOT training data — which is
# deliberately excluded here and in scripts/inspect_dataset.py).
# Class_0 = no oil spill, Class_1 = oil spill (confirmed with project owner
# and matches the CSIRO metadata description above).
CLASS_MAP = {
    "Class_0": 0,  # no oil (look-alikes or clean seas)
    "Class_1": 1,  # oil spill features
}
CLASS_NAMES = ["no_oil", "oil"]  # index-aligned with the integer labels above

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

for d in (MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Data / training settings ---------------------------------------------

IMAGE_SIZE = 224          # ResNet18 standard input size
BATCH_SIZE = 32
# 0 = load data in the main process. On Windows, multiprocess DataLoader
# workers (num_workers>0) use the "spawn" start method, which can hang when
# launched from certain non-interactive shells. 0 is slightly slower per
# batch but reliable, and this dataset is small enough that it's not a
# meaningful bottleneck.
NUM_WORKERS = 0

# Stratified split ratios (must sum to 1.0)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# Optional cap on the TRAINING split only (val/test are always used in full,
# since honest evaluation needs the full held-out data). CPU-only training
# on the full ~3,900-image train split takes ~25 min/epoch on this machine,
# which is impractical. Subsampling (stratified, so the class ratio is
# preserved) trades some training data for speed. Set to None to disable
# and use the full training split.
TRAIN_SUBSET_SIZE = 1800

# --- Model / training hyperparameters --------------------------------------

NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5   # stop if val balanced accuracy doesn't improve for N epochs

MODEL_PATH = MODELS_DIR / "oil_spill_resnet18.pt"
TRAINING_CURVE_PATH = OUTPUTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
METRICS_REPORT_PATH = OUTPUTS_DIR / "metrics_report.txt"
