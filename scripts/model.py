"""
TreeVisionAI — EfficientNet-B0 Model
=====================================

EfficientNet-B0 model builder for leaf species classification.

Features:
  - Loads pretrained ImageNet weights automatically
  - Replaces classifier head for NUM_CLASSES outputs
  - freeze_backbone() / unfreeze_backbone() for two-phase training
  - save_checkpoint() / load_checkpoint() with full training state
  - Scalable: num_classes is configurable

Usage:
    from scripts.model import build_model, save_checkpoint, load_checkpoint

    model = build_model(num_classes=4, pretrained=True)
"""

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    NUM_CLASSES,
    SPECIES_LIST,
    SPECIES,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    MODELS_DIR,
    LOGS_DIR,
    setup_logging,
    get_device,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="model",
    log_file=os.path.join(LOGS_DIR, "model.log"),
)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an EfficientNet-B0 model for species classification.

    Loads ImageNet pretrained weights and replaces the
    final classifier layer with a new head for `num_classes`.

    Architecture:
        EfficientNet-B0 backbone (frozen initially)
        → AdaptiveAvgPool2d
        → Dropout(0.2)
        → Linear(1280 → num_classes)

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        EfficientNet-B0 model with modified classifier.
    """

    if pretrained:
        logger.info("Loading EfficientNet-B0 with ImageNet weights...")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        logger.info("Loading EfficientNet-B0 without pretrained weights...")
        model = models.efficientnet_b0(weights=None)

    # Get the number of input features to the classifier
    in_features = model.classifier[1].in_features

    # Replace the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    logger.info(
        f"  Classifier head replaced: "
        f"Linear({in_features} → {num_classes})"
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"  Total parameters     : {total_params:,}"
        f"\n  Trainable parameters : {trainable_params:,}"
    )

    return model


# ============================================================
# FREEZE / UNFREEZE BACKBONE
# ============================================================

def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all backbone (feature extractor) layers.
    Only the classifier head will be trainable.

    Used in Phase 1 training.
    """

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"  Backbone FROZEN. Trainable params: {trainable:,}"
    )


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all layers for full fine-tuning.

    Used in Phase 2 training.
    """

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"  Backbone UNFROZEN. Trainable params: {trainable:,}"
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
    class_names: List[str],
    filepath: str,
    scheduler: Optional[object] = None,
) -> None:
    """
    Save a full training checkpoint.

    Includes model weights, optimizer state, training metrics,
    and class names for reproducibility.

    Args:
        model: The model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        train_loss: Training loss.
        val_loss: Validation loss.
        train_accuracy: Training accuracy (0-100).
        val_accuracy: Validation accuracy (0-100).
        class_names: List of class names.
        filepath: Path to save the checkpoint.
        scheduler: Optional LR scheduler state.
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "class_names": class_names,
        "num_classes": len(class_names),
        "architecture": "EfficientNet-B0",
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)

    logger.info(
        f"  Checkpoint saved: {os.path.basename(filepath)} "
        f"(epoch={epoch}, val_acc={val_accuracy:.2f}%)"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load a saved checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to map the checkpoint to.

    Returns:
        Dict with checkpoint metadata (epoch, accuracy, etc.)
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"  Checkpoint loaded: {os.path.basename(filepath)}\n"
        f"    Epoch        : {checkpoint['epoch']}\n"
        f"    Val Accuracy : {checkpoint['val_accuracy']:.2f}%\n"
        f"    Classes      : {checkpoint['class_names']}"
    )

    return checkpoint


# ============================================================
# MAIN — Test model build
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Model Builder Test")
    print("=" * 60 + "\n")

    device = get_device()

    # Build model
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    print(f"\n  Device   : {device}")
    print(f"  Classes  : {NUM_CLASSES} ({SPECIES_LIST})")

    # Test freeze/unfreeze
    print("\n  --- Phase 1: Freeze backbone ---")
    freeze_backbone(model)

    print("\n  --- Phase 2: Unfreeze backbone ---")
    unfreeze_backbone(model)

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\n  Input  shape : {dummy_input.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output       : {output.detach().cpu().numpy()}")

    print(f"\n  [OK] Model built and tested successfully!\n")
