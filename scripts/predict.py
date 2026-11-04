"""
TreeVisionAI — Single Image Prediction
========================================

Predict tree species from a single leaf image.

Loads the best trained model and returns the
predicted species with confidence score.

Usage:
    python scripts/predict.py --image path/to/leaf.jpg
    python scripts/predict.py --image path/to/leaf.jpg --top 3
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES,
    BEST_MODEL_PATH,
    LOGS_DIR,
    setup_logging,
    get_device,
)

from scripts.model import build_model, load_checkpoint
from scripts.dataset import get_val_transforms


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="predictor",
    log_file=os.path.join(LOGS_DIR, "predict.log"),
)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image_path: str,
    model_path: str = BEST_MODEL_PATH,
    top_k: int = 4,
) -> List[Dict]:
    """
    Predict the species of a leaf image.

    Args:
        image_path: Path to the input leaf image.
        model_path: Path to the trained model checkpoint.
        top_k: Number of top predictions to return.

    Returns:
        List of dicts with species, confidence, display_name, scientific_name.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the model first with: python scripts/train.py"
        )

    device = get_device()

    # Load checkpoint to get class names
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    # Build and load model
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load and transform image
    transform = get_val_transforms()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, num_classes))

    # Format results
    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):

        species_key = class_names[idx.item()]
        species_info = SPECIES.get(species_key, {})

        results.append({
            "species_key": species_key,
            "display_name": species_info.get("display_name", species_key),
            "scientific_name": species_info.get("scientific_name", "Unknown"),
            "confidence": round(prob.item() * 100, 2),
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main(
    image_path: str,
    model_path: str = BEST_MODEL_PATH,
    top_k: int = 4,
) -> None:
    """
    Predict species and print results.

    Args:
        image_path: Path to the input leaf image.
        model_path: Path to the model checkpoint.
        top_k: Number of top predictions to show.
    """

    print(f"\n{'=' * 60}")
    print(f"  TreeVisionAI — Species Prediction")
    print(f"{'=' * 60}")
    print(f"\n  Image : {image_path}")
    print(f"  Model : {os.path.basename(model_path)}\n")

    results = predict_image(image_path, model_path, top_k)

    print(f"  {'Rank':<6} {'Species':<20} {'Scientific Name':<30} {'Confidence':>10}")
    print(f"  {'---':<6} {'---':<20} {'---':<30} {'---':>10}")

    for i, r in enumerate(results, 1):
        conf_str = f"{r['confidence']:.2f}%"
        print(
            f"  {i:<6} {r['display_name']:<20} "
            f"{r['scientific_name']:<30} {conf_str:>10}"
        )

    # Highlight top prediction
    top = results[0]
    print(f"\n  >> Predicted: {top['display_name']} "
          f"({top['scientific_name']}) "
          f"with {top['confidence']:.2f}% confidence")

    print(f"\n{'=' * 60}\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Predict tree species from leaf image"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the leaf image",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_PATH,
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=4,
        help="Number of top predictions to show (default: 4)",
    )

    args = parser.parse_args()

    main(
        image_path=args.image,
        model_path=args.model,
        top_k=args.top,
    )
