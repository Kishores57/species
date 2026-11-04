"""
TreeVisionAI — PyTorch Dataset & DataLoaders
=============================================

Custom Dataset class and DataLoader factory for leaf
image classification. Includes:
  - Train augmentations (flip, rotate, crop, perspective, color jitter)
  - Val/Test transforms (resize + center crop)
  - ImageNet normalization
  - Class-balanced WeightedRandomSampler for imbalanced classes

Usage:
    from scripts.dataset import get_dataloaders

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
"""

import os
import sys
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    SPECIES_LIST,
    LOGS_DIR,
    setup_logging,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="dataset",
    log_file=os.path.join(LOGS_DIR, "dataset.log"),
)


# ============================================================
# DATA TRANSFORMS
# ============================================================

def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Applies aggressive augmentation to improve generalization:
    - RandomResizedCrop: random scale + crop to image_size
    - RandomHorizontalFlip: 50% chance
    - RandomRotation: ±15 degrees
    - RandomPerspective: subtle perspective warp
    - ColorJitter: brightness, contrast, saturation, hue
    - RandomAffine: small translations + scale changes
    - Normalize: ImageNet mean/std
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Validation / Test transforms.

    No augmentation — only resize, center crop, and normalize.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# WEIGHTED SAMPLER FOR CLASS IMBALANCE
# ============================================================

def get_weighted_sampler(
    dataset: datasets.ImageFolder,
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.

    Oversamples underrepresented classes so each class is
    seen equally often per epoch.

    Args:
        dataset: ImageFolder dataset.

    Returns:
        WeightedRandomSampler instance.
    """

    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)

    # Weight = 1 / count (inverse frequency)
    class_weights = 1.0 / class_counts.float()

    # Assign weight to each sample based on its class
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


# ============================================================
# GET DATALOADERS
# ============================================================

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMAGE_SIZE,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test DataLoaders.

    Uses torchvision.datasets.ImageFolder which expects:
        datasets/species/train/<class_name>/image.jpg
        datasets/species/val/<class_name>/image.jpg
        datasets/species/test/<class_name>/image.jpg

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        image_size: Input image size.
        use_weighted_sampler: Whether to use WeightedRandomSampler
            for training (handles class imbalance).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """

    train_dir = os.path.join(SPECIES_DIR, "train")
    val_dir = os.path.join(SPECIES_DIR, "val")
    test_dir = os.path.join(SPECIES_DIR, "test")

    # Verify directories exist
    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"Dataset directory not found: {d}\n"
                f"Run split_dataset.py first."
            )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_train_transforms(image_size),
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_val_transforms(image_size),
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=get_val_transforms(image_size),
    )

    class_names = train_dataset.classes

    # Log dataset info
    logger.info(
        f"\n  Dataset loaded:"
        f"\n    Train : {len(train_dataset)} images"
        f"\n    Val   : {len(val_dataset)} images"
        f"\n    Test  : {len(test_dataset)} images"
        f"\n    Classes: {class_names}"
    )

    # Class distribution
    train_targets = torch.tensor(train_dataset.targets)
    for idx, name in enumerate(class_names):
        count = (train_targets == idx).sum().item()
        logger.info(f"    Train [{name}]: {count} images")

    # Create DataLoaders
    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler:
        train_sampler = get_weighted_sampler(train_dataset)
        train_shuffle = False  # Sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# MAIN — Test DataLoaders
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Dataset & DataLoader Test")
    print("=" * 60 + "\n")

    try:
        train_loader, val_loader, test_loader, class_names = get_dataloaders()

        print(f"  Classes : {class_names}")
        print(f"  Train batches : {len(train_loader)}")
        print(f"  Val batches   : {len(val_loader)}")
        print(f"  Test batches  : {len(test_loader)}")

        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"\n  Sample batch shape  : {images.shape}")
        print(f"  Sample labels       : {labels[:8].tolist()}")
        print(f"  Label names         : {[class_names[l] for l in labels[:8].tolist()]}")

        print(f"\n  [OK] DataLoaders working correctly!\n")

    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print(f"  Run split_dataset.py first.\n")
