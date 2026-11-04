"""
TreeVisionAI — Dataset Splitter
================================

Splits cleaned images into train/val/test sets with
stratified random sampling.

Source: datasets/cleaned/<species>/
Target: datasets/species/train/<species>/
        datasets/species/val/<species>/
        datasets/species/test/<species>/

Ratio: 80% train / 10% val / 10% test (configurable)

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --ratio 0.7 0.15 0.15
"""

import os
import sys
import shutil
import random
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES,
    SPECIES_LIST,
    CLEANED_DIR,
    SPECIES_DIR,
    LOGS_DIR,
    IMAGE_EXTENSIONS,
    SPLIT_RATIOS,
    SPLIT_SEED,
    setup_logging,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="splitter",
    log_file=os.path.join(LOGS_DIR, "split_dataset.log"),
)


# ============================================================
# SPLIT SINGLE SPECIES
# ============================================================

def split_species(
    species_key: str,
    ratios: Dict[str, float],
    seed: int = SPLIT_SEED,
) -> Dict[str, int]:
    """
    Split a single species into train/val/test.

    Uses stratified random sampling with a fixed seed
    for reproducibility.

    Args:
        species_key: Species folder key.
        ratios: Dict with split names and ratios.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name → count.
    """

    display_name = SPECIES[species_key]["display_name"]
    source_dir = os.path.join(CLEANED_DIR, species_key)

    # Collect all image filenames
    if not os.path.exists(source_dir):
        logger.warning(f"Source not found: {source_dir}")
        return {}

    images = sorted([
        f for f in os.listdir(source_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not images:
        logger.warning(f"No images found for {display_name}.")
        return {}

    total = len(images)
    logger.info(f"  {display_name}: {total} images")

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(images)

    # Compute split sizes
    split_names = list(ratios.keys())
    split_sizes = {}
    remaining = total

    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            # Last split gets all remaining
            split_sizes[name] = remaining
        else:
            size = int(total * ratios[name])
            split_sizes[name] = size
            remaining -= size

    # Assign images to splits
    idx = 0
    split_counts = {}

    for split_name in split_names:

        size = split_sizes[split_name]
        split_images = images[idx:idx + size]
        idx += size

        # Create target directory
        target_dir = os.path.join(SPECIES_DIR, split_name, species_key)
        os.makedirs(target_dir, exist_ok=True)

        # Copy images
        for filename in split_images:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        split_counts[split_name] = len(split_images)

    return split_counts


# ============================================================
# MAIN
# ============================================================

def main(
    ratios: Dict[str, float] = None,
    seed: int = SPLIT_SEED,
) -> None:
    """
    Split all species into train/val/test.

    Args:
        ratios: Split ratios. None = use config defaults.
        seed: Random seed.
    """

    if ratios is None:
        ratios = SPLIT_RATIOS

    # Validate ratios sum to 1.0
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(
            f"Split ratios must sum to 1.0. Got {total_ratio:.2f}"
        )
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  TreeVisionAI — Dataset Splitter")
    print(f"{'=' * 60}")
    print(f"\n  Ratios: {ratios}")
    print(f"  Seed  : {seed}")
    print(f"  Source : {CLEANED_DIR}")
    print(f"  Target : {SPECIES_DIR}\n")

    all_counts = {}

    for species_key in SPECIES_LIST:
        counts = split_species(species_key, ratios, seed)
        all_counts[species_key] = counts

    # Print summary table
    print(f"\n  {'Species':<12}", end="")
    for split_name in ratios:
        print(f"  {split_name.capitalize():>8}", end="")
    print(f"  {'Total':>8}")
    print(f"  {'-' * (12 + 10 * (len(ratios) + 1))}")

    for species_key in SPECIES_LIST:
        name = SPECIES[species_key]["display_name"]
        counts = all_counts.get(species_key, {})
        total = sum(counts.values())

        print(f"  {name:<12}", end="")
        for split_name in ratios:
            print(f"  {counts.get(split_name, 0):>8}", end="")
        print(f"  {total:>8}")

    # Grand total
    print(f"  {'-' * (12 + 10 * (len(ratios) + 1))}")
    print(f"  {'TOTAL':<12}", end="")
    for split_name in ratios:
        col_total = sum(
            all_counts.get(k, {}).get(split_name, 0)
            for k in SPECIES_LIST
        )
        print(f"  {col_total:>8}", end="")
    grand_total = sum(
        sum(c.values()) for c in all_counts.values()
    )
    print(f"  {grand_total:>8}")

    print(f"\n{'=' * 60}\n")

    logger.info(f"Dataset split complete. Total: {grand_total} images.")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Split dataset into train/val/test"
    )

    parser.add_argument(
        "--ratio",
        nargs=3,
        type=float,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum to 1.0). Default: 0.8 0.1 0.1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=f"Random seed (default: {SPLIT_SEED})",
    )

    args = parser.parse_args()

    if args.ratio:
        ratios = {
            "train": args.ratio[0],
            "val": args.ratio[1],
            "test": args.ratio[2],
        }
    else:
        ratios = None

    main(ratios=ratios, seed=args.seed)
