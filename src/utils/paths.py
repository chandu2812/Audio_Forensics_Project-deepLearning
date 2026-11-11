"""Centralized path management for the project."""
from pathlib import Path
import os

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

ASVSPOOF_RAW = RAW_DIR / "ASVspoof"
WAVEFAKE_RAW = RAW_DIR / "WaveFake"

AUDIO_PROCESSED = PROCESSED_DIR / "audio"
SPECTROGRAM_PROCESSED = PROCESSED_DIR / "spectrograms"

# CSV files
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CNN_BEST_MODEL = MODELS_DIR / "cnn_best.pth"
TRANSFORMER_BEST_MODEL = MODELS_DIR / "transformer_best.pth"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
CONFUSION_DIR = OUTPUTS_DIR / "confusion_matrices"
ROC_DIR = OUTPUTS_DIR / "roc_curves"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for path in [
        PROCESSED_DIR, AUDIO_PROCESSED, SPECTROGRAM_PROCESSED,
        METADATA_DIR, MODELS_DIR, GRAPHS_DIR,
        CONFUSION_DIR, ROC_DIR, REPORTS_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)