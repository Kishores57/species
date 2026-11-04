"""
TreeVisionAI — Project Configuration
=====================================

Single source of truth for all paths, species definitions,
image quality thresholds, training hyperparameters, and
device configuration.

Every other script imports from this module.
No hardcoded paths or magic numbers anywhere else.
"""

import os
import logging
from typing import Dict, List, Tuple


# ============================================================
# PROJECT ROOT
# ============================================================

# Root is the parent of the 'scripts/' directory
PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

DATASETS_DIR: str = os.path.join(PROJECT_ROOT, "datasets")
RAW_DIR: str = os.path.join(DATASETS_DIR, "raw")
CLEANED_DIR: str = os.path.join(DATASETS_DIR, "cleaned")
REVIEW_DIR: str = os.path.join(DATASETS_DIR, "review")
SPECIES_DIR: str = os.path.join(DATASETS_DIR, "species")
REPORTS_DIR: str = os.path.join(DATASETS_DIR, "reports")

MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")
SCRIPTS_DIR: str = os.path.join(PROJECT_ROOT, "scripts")


# ============================================================
# TARGET SPECIES
# ============================================================

# Each species has a folder key, display name, and scientific name.
# To add a new species, just add an entry here.
# The entire pipeline adapts automatically.

SPECIES: Dict[str, Dict[str, str]] = {
    "neem": {
        "display_name": "Neem",
        "scientific_name": "Azadirachta indica",
    },
    "raintree": {
        "display_name": "Rain Tree",
        "scientific_name": "Samanea saman",
    },
    "mango": {
        "display_name": "Mango",
        "scientific_name": "Mangifera indica",
    },
    "ashoka": {
        "display_name": "Ashoka",
        "scientific_name": "Polyalthia longifolia",
    },
    "peepal": {
        "display_name": "Peepal",
        "scientific_name": "Ficus religiosa",
    },
    "rubber": {
        "display_name": "Rubber Tree",
        "scientific_name": "Ficus elastica",
    },
    "banyan": {
        "display_name": "Banyan",
        "scientific_name": "Ficus benghalensis",
    },
}

# Ordered list of species keys (used for class index mapping)
SPECIES_LIST: List[str] = sorted(SPECIES.keys())

# Number of classes
NUM_CLASSES: int = len(SPECIES_LIST)

# Class index mapping: {"ashoka": 0, "mango": 1, "neem": 2, "raintree": 3}
CLASS_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(SPECIES_LIST)
}

IDX_TO_CLASS: Dict[int, str] = {
    idx: name for name, idx in CLASS_TO_IDX.items()
}


# ============================================================
# DATASET DOWNLOAD
# ============================================================

DOWNLOAD_TARGET_PER_SPECIES: int = 3000  # ~2k training imgs after clean + 80/10/10 split
DOWNLOAD_THREADS: int = 16
DOWNLOAD_TIMEOUT: int = 30               # Seconds per image request
DOWNLOAD_MAX_RETRIES: int = 3
DOWNLOAD_PER_PAGE: int = 200             # iNaturalist API max

# iNaturalist API
INAT_API_URL: str = "https://api.inaturalist.org/v1/observations"
INAT_USER_AGENT: str = "TreeVisionAI/1.0 (research project)"


# ============================================================
# IMAGE QUALITY THRESHOLDS
# ============================================================

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

MIN_WIDTH: int = 224           # Minimum image width in pixels
MIN_HEIGHT: int = 224          # Minimum image height in pixels

BLUR_THRESHOLD: float = 100.0  # Laplacian variance below this = blurry
BRIGHTNESS_MIN: int = 30       # Mean pixel value below this = too dark
BRIGHTNESS_MAX: int = 240      # Mean pixel value above this = overexposed

# Duplicate detection
PHASH_THRESHOLD: int = 5       # Perceptual hash distance <= this = duplicate


# ============================================================
# DATASET SPLIT
# ============================================================

SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

SPLIT_SEED: int = 42  # Reproducible splits


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

# Input image size for EfficientNet-B0
IMAGE_SIZE: int = 224

# ImageNet normalization constants
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Batch size — tuned for RTX 2050 (4GB VRAM)
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4

# Phase 1: Train classifier head only
PHASE1_EPOCHS: int = 10
PHASE1_LR: float = 1e-3

# Phase 2: Fine-tune entire model
PHASE2_EPOCHS: int = 30
PHASE2_LR: float = 1e-4

# Optimizer
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTHING: float = 0.1

# Scheduler
SCHEDULER_T_MAX: int = PHASE2_EPOCHS  # For CosineAnnealingLR

# Early stopping
EARLY_STOPPING_PATIENCE: int = 7

# Gradient clipping
MAX_GRAD_NORM: float = 1.0


# ============================================================
# MODEL PATHS
# ============================================================

BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "best_model.pth")
LAST_MODEL_PATH: str = os.path.join(MODELS_DIR, "last_model.pth")

# Training history
TRAINING_HISTORY_PATH: str = os.path.join(LOGS_DIR, "training_history.csv")

# Evaluation outputs
CONFUSION_MATRIX_PATH: str = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
ACCURACY_CURVE_PATH: str = os.path.join(OUTPUTS_DIR, "accuracy_curve.png")
LOSS_CURVE_PATH: str = os.path.join(OUTPUTS_DIR, "loss_curve.png")


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device() -> str:
    """
    Detect the best available device.

    Returns:
        'cuda' if NVIDIA GPU with CUDA is available,
        'cpu' otherwise.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
            )
            logging.info(f"GPU detected: {gpu_name} ({vram_gb} GB VRAM)")
            return "cuda"
        else:
            logging.warning(
                "CUDA not available. Training will use CPU (much slower)."
            )
            return "cpu"

    except ImportError:
        logging.warning("PyTorch not installed. Device set to CPU.")
        return "cpu"


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(
    name: str = "TreeVisionAI",
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with console + optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# CREATE ALL DIRECTORIES
# ============================================================

def create_project_dirs() -> None:
    """Create all required project directories."""

    dirs = [
        RAW_DIR,
        CLEANED_DIR,
        REVIEW_DIR,
        SPECIES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
    ]

    # Create per-species subdirectories
    for species_key in SPECIES:
        dirs.append(os.path.join(RAW_DIR, species_key))
        dirs.append(os.path.join(CLEANED_DIR, species_key))
        dirs.append(os.path.join(REVIEW_DIR, species_key))

    # Create split directories
    for split in SPLIT_RATIOS:
        for species_key in SPECIES:
            dirs.append(
                os.path.join(SPECIES_DIR, split, species_key)
            )

    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ============================================================
# PRINT CONFIGURATION SUMMARY
# ============================================================

def print_config() -> None:
    """Print a human-readable summary of the current configuration."""

    device = get_device()

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Configuration Summary")
    print("=" * 60)

    print(f"\n  Project Root   : {PROJECT_ROOT}")
    print(f"  Device         : {device.upper()}")
    print(f"  Num Classes    : {NUM_CLASSES}")

    print("\n  Species:")
    for key, info in SPECIES.items():
        idx = CLASS_TO_IDX[key]
        print(f"    [{idx}] {info['display_name']} ({info['scientific_name']})")

    print(f"\n  Image Size     : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch Size     : {BATCH_SIZE}")
    print(f"  Phase 1        : {PHASE1_EPOCHS} epochs, lr={PHASE1_LR}")
    print(f"  Phase 2        : {PHASE2_EPOCHS} epochs, lr={PHASE2_LR}")
    print(f"  Early Stopping : patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Split Ratio    : {SPLIT_RATIOS}")

    print("\n" + "=" * 60 + "\n")


# ============================================================
# AUTO-CREATE DIRECTORIES ON IMPORT
# ============================================================

create_project_dirs()


# ============================================================
# MAIN — Run to verify configuration
# ============================================================

if __name__ == "__main__":
    print_config()