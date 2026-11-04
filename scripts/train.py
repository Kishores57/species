"""
TreeVisionAI — Training Engine
================================

Two-phase training pipeline for EfficientNet-B0:

Phase 1: Freeze backbone, train classifier head only
  - Epochs: 5-10
  - LR: 1e-3 (AdamW)

Phase 2: Unfreeze backbone, fine-tune entire model
  - Epochs: up to 30
  - LR: 1e-4 (AdamW)
  - Scheduler: CosineAnnealingLR
  - Early Stopping: patience=7

Saves:
  - models/best_model.pth  (highest val accuracy)
  - models/last_model.pth  (final epoch)
  - logs/training_history.csv

Usage:
    python scripts/train.py
    python scripts/train.py --phase1-epochs 5 --phase2-epochs 20
"""

import os
import sys
import csv
import time
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    SCHEDULER_T_MAX,
    EARLY_STOPPING_PATIENCE,
    MAX_GRAD_NORM,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    LOGS_DIR,
    setup_logging,
    get_device,
)

from scripts.model import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
    save_checkpoint,
)

from scripts.dataset import get_dataloaders


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="trainer",
    log_file=os.path.join(LOGS_DIR, "training.log"),
)


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement to reset patience counter.
    """

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device ('cuda' or 'cpu').
        scaler: GradScaler for mixed precision (optional).
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="    Train", leave=False, unit="batch")

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None and device == "cuda":
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# VALIDATE ONE EPOCH
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Evaluate the model on validation data.

    Args:
        model: The model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="    Val  ", leave=False, unit="batch")

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# TRAINING HISTORY
# ============================================================

class TrainingHistory:
    """Records and saves training metrics per epoch."""

    def __init__(self, filepath: str = TRAINING_HISTORY_PATH):
        self.filepath = filepath
        self.records: List[Dict] = []

        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def add(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Add a record for one epoch."""
        self.records.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_acc, 2),
            "val_accuracy": round(val_acc, 2),
            "learning_rate": lr,
        })

    def save(self) -> None:
        """Save all records to CSV."""
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "phase", "train_loss", "val_loss",
                "train_accuracy", "val_accuracy", "learning_rate",
            ])
            writer.writeheader()
            writer.writerows(self.records)

        logger.info(f"  Training history saved: {self.filepath}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
    phase1_lr: float = PHASE1_LR,
    phase2_lr: float = PHASE2_LR,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Full two-phase training pipeline.

    Phase 1: Freeze backbone, train classifier head.
    Phase 2: Unfreeze all, fine-tune with low LR.

    Args:
        phase1_epochs: Number of epochs for Phase 1.
        phase2_epochs: Max epochs for Phase 2.
        phase1_lr: Learning rate for Phase 1.
        phase2_lr: Learning rate for Phase 2.
        batch_size: Batch size.
    """

    device = get_device()

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Training Pipeline")
    print("=" * 60)
    print(f"\n  Device       : {device.upper()}")
    print(f"  Phase 1      : {phase1_epochs} epochs, lr={phase1_lr}")
    print(f"  Phase 2      : {phase2_epochs} epochs, lr={phase2_lr}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Early Stop   : patience={EARLY_STOPPING_PATIENCE}")

    # --------------------------------
    # Load data
    # --------------------------------
    logger.info("Loading datasets...")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=batch_size,
    )

    print(f"  Classes      : {class_names}")
    print(f"  Train images : {len(train_loader.dataset)}")
    print(f"  Val images   : {len(val_loader.dataset)}")

    # --------------------------------
    # Build model
    # --------------------------------
    model = build_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Mixed precision scaler
    scaler = GradScaler("cuda") if device == "cuda" else None

    # Training history
    history = TrainingHistory()

    # Best validation accuracy tracker
    best_val_acc = 0.0
    global_epoch = 0

    # ================================
    # PHASE 1: Train Classifier Head
    # ================================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1 — Train Classifier Head Only")
    print(f"{'=' * 60}\n")

    freeze_backbone(model)

    # Only optimize classifier parameters
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(1, phase1_epochs + 1):

        global_epoch += 1
        logger.info(f"  Phase 1 | Epoch {epoch}/{phase1_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer1, device, scaler
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer1.param_groups[0]["lr"]

        history.add(
            epoch=global_epoch, phase="phase1",
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc,
            lr=current_lr,
        )

        logger.info(
            f"    Loss: {train_loss:.4f} / {val_loss:.4f} | "
            f"Acc: {train_acc:.2f}% / {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer1, global_epoch,
                train_loss, val_loss, train_acc, val_acc,
                class_names, BEST_MODEL_PATH,
            )

    # Save last model after Phase 1
    save_checkpoint(
        model, optimizer1, global_epoch,
        train_loss, val_loss, train_acc, val_acc,
        class_names, LAST_MODEL_PATH,
    )

    # ================================
    # PHASE 2: Fine-tune Entire Model
    # ================================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2 — Fine-tune Entire Model")
    print(f"{'=' * 60}\n")

    unfreeze_backbone(model)

    # Optimizer for all parameters with low LR
    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=phase2_lr,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=phase2_epochs,
        eta_min=1e-6,
    )

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    for epoch in range(1, phase2_epochs + 1):

        global_epoch += 1
        logger.info(f"  Phase 2 | Epoch {epoch}/{phase2_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer2, device, scaler
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer2.param_groups[0]["lr"]

        history.add(
            epoch=global_epoch, phase="phase2",
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc,
            lr=current_lr,
        )

        logger.info(
            f"    Loss: {train_loss:.4f} / {val_loss:.4f} | "
            f"Acc: {train_acc:.2f}% / {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer2, global_epoch,
                train_loss, val_loss, train_acc, val_acc,
                class_names, BEST_MODEL_PATH, scheduler,
            )

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(
                f"\n  Early stopping triggered at epoch {epoch}. "
                f"Best val_acc: {best_val_acc:.2f}%"
            )
            break

    # Save last model after Phase 2
    save_checkpoint(
        model, optimizer2, global_epoch,
        train_loss, val_loss, train_acc, val_acc,
        class_names, LAST_MODEL_PATH, scheduler,
    )

    # Save training history
    history.save()

    # --------------------------------
    # Final Summary
    # --------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"{'=' * 60}")
    print(f"\n  Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"  Total Epochs Trained     : {global_epoch}")
    print(f"  Best Model Saved         : {BEST_MODEL_PATH}")
    print(f"  Last Model Saved         : {LAST_MODEL_PATH}")
    print(f"  Training History         : {TRAINING_HISTORY_PATH}")
    print(f"\n{'=' * 60}\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Train EfficientNet-B0"
    )

    parser.add_argument(
        "--phase1-epochs", type=int, default=PHASE1_EPOCHS,
        help=f"Phase 1 epochs (default: {PHASE1_EPOCHS})",
    )
    parser.add_argument(
        "--phase2-epochs", type=int, default=PHASE2_EPOCHS,
        help=f"Phase 2 max epochs (default: {PHASE2_EPOCHS})",
    )
    parser.add_argument(
        "--phase1-lr", type=float, default=PHASE1_LR,
        help=f"Phase 1 learning rate (default: {PHASE1_LR})",
    )
    parser.add_argument(
        "--phase2-lr", type=float, default=PHASE2_LR,
        help=f"Phase 2 learning rate (default: {PHASE2_LR})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )

    args = parser.parse_args()

    train(
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        batch_size=args.batch_size,
    )
