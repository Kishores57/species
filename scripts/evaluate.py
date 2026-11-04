"""
TreeVisionAI — Model Evaluation
=================================

Evaluates the best trained model on the test set.

Generates:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix heatmap → outputs/confusion_matrix.png
  - Accuracy curves → outputs/accuracy_curve.png
  - Loss curves → outputs/loss_curve.png

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/best_model.pth
"""

import os
import sys
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    NUM_CLASSES,
    SPECIES,
    SPECIES_LIST,
    BEST_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    CONFUSION_MATRIX_PATH,
    ACCURACY_CURVE_PATH,
    LOSS_CURVE_PATH,
    OUTPUTS_DIR,
    LOGS_DIR,
    setup_logging,
    get_device,
)

from scripts.model import build_model, load_checkpoint
from scripts.dataset import get_dataloaders


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="evaluator",
    log_file=os.path.join(LOGS_DIR, "evaluate.log"),
)


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Get all predictions and true labels from the test set.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        device: Device.

    Returns:
        Tuple of (true_labels, predicted_labels, confidences).
    """

    model.eval()

    all_labels = []
    all_preds = []
    all_confs = []

    pbar = tqdm(dataloader, desc="  Evaluating", unit="batch")

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(predictions.cpu().numpy().tolist())
        all_confs.extend(confidences.cpu().numpy().tolist())

    return all_labels, all_preds, all_confs


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    save_path: str = CONFUSION_MATRIX_PATH,
) -> None:
    """
    Generate and save a confusion matrix heatmap.
    """

    cm = confusion_matrix(true_labels, pred_labels)

    # Display names
    display_names = [
        SPECIES.get(name, {}).get("display_name", name)
        for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("TreeVisionAI - Confusion Matrix", fontsize=14, pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  Confusion matrix saved: {save_path}")


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(
    history_path: str = TRAINING_HISTORY_PATH,
    acc_save_path: str = ACCURACY_CURVE_PATH,
    loss_save_path: str = LOSS_CURVE_PATH,
) -> None:
    """
    Plot accuracy and loss curves from training history CSV.
    """

    if not os.path.exists(history_path):
        logger.warning(f"Training history not found: {history_path}")
        return

    df = pd.read_csv(history_path)

    os.makedirs(os.path.dirname(acc_save_path), exist_ok=True)

    # ---- Accuracy Curve ----
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["epoch"], df["train_accuracy"], "b-o",
            label="Train Accuracy", markersize=4)
    ax.plot(df["epoch"], df["val_accuracy"], "r-o",
            label="Val Accuracy", markersize=4)

    # Mark phase boundary
    phase1 = df[df["phase"] == "phase1"]
    phase2 = df[df["phase"] == "phase2"]
    if len(phase1) > 0 and len(phase2) > 0:
        boundary = phase1["epoch"].max() + 0.5
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   alpha=0.7, label="Phase 1 / Phase 2")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("TreeVisionAI - Training Accuracy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(acc_save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Accuracy curve saved: {acc_save_path}")

    # ---- Loss Curve ----
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["epoch"], df["train_loss"], "b-o",
            label="Train Loss", markersize=4)
    ax.plot(df["epoch"], df["val_loss"], "r-o",
            label="Val Loss", markersize=4)

    if len(phase1) > 0 and len(phase2) > 0:
        boundary = phase1["epoch"].max() + 0.5
        ax.axvline(x=boundary, color="gray", linestyle="--",
                   alpha=0.7, label="Phase 1 / Phase 2")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("TreeVisionAI - Training Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(loss_save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Loss curve saved: {loss_save_path}")


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(model_path: str = BEST_MODEL_PATH) -> None:
    """
    Full evaluation pipeline.

    1. Load best model
    2. Run on test set
    3. Print classification report
    4. Generate confusion matrix
    5. Plot training curves

    Args:
        model_path: Path to the saved model checkpoint.
    """

    device = get_device()

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Model Evaluation")
    print("=" * 60)
    print(f"\n  Model  : {model_path}")
    print(f"  Device : {device.upper()}\n")

    # Load data
    _, _, test_loader, class_names = get_dataloaders()

    print(f"  Test images : {len(test_loader.dataset)}")
    print(f"  Classes     : {class_names}\n")

    # Build and load model
    model = build_model(num_classes=len(class_names), pretrained=False)
    checkpoint = load_checkpoint(model_path, model, device=device)
    model = model.to(device)

    # Get predictions
    true_labels, pred_labels, confidences = get_predictions(
        model, test_loader, device
    )

    # Display names for the report
    display_names = [
        SPECIES.get(name, {}).get("display_name", name)
        for name in class_names
    ]

    # ---- Classification Report ----
    print(f"\n{'=' * 60}")
    print("  Classification Report")
    print(f"{'=' * 60}\n")

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=display_names,
        digits=4,
    )
    print(report)

    # Overall accuracy
    overall_acc = accuracy_score(true_labels, pred_labels) * 100
    avg_confidence = sum(confidences) / len(confidences) * 100

    print(f"  Overall Test Accuracy  : {overall_acc:.2f}%")
    print(f"  Average Confidence     : {avg_confidence:.2f}%")

    # ---- Confusion Matrix ----
    plot_confusion_matrix(true_labels, pred_labels, class_names)

    # ---- Training Curves ----
    plot_training_curves()

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  Evaluation Complete!")
    print(f"{'=' * 60}")
    print(f"\n  Test Accuracy     : {overall_acc:.2f}%")
    print(f"  Confusion Matrix  : {CONFUSION_MATRIX_PATH}")
    print(f"  Accuracy Curve    : {ACCURACY_CURVE_PATH}")
    print(f"  Loss Curve        : {LOSS_CURVE_PATH}")
    print(f"\n{'=' * 60}\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Evaluate trained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_PATH,
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )

    args = parser.parse_args()

    evaluate(model_path=args.model)
