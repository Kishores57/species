"""
TreeVisionAI — Dataset Verifier
================================

Post-cleaning verification script.
Validates all cleaned images and generates a summary report.

Checks:
  - Image count per species
  - Class balance / imbalance warnings
  - All images are loadable (PIL + OpenCV)
  - Min/Max/Avg resolution per species
  - Cross-species duplicates (same image in two species)
  - File integrity

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --source cleaned
    python scripts/verify_dataset.py --source species
"""

import os
import sys
import csv
import hashlib
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES,
    SPECIES_LIST,
    CLEANED_DIR,
    SPECIES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    IMAGE_EXTENSIONS,
    setup_logging,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="verifier",
    log_file=os.path.join(LOGS_DIR, "verify_dataset.log"),
)


# ============================================================
# SCAN SINGLE IMAGE
# ============================================================

def scan_image(filepath: str) -> Dict:
    """
    Verify a single image is valid and get its stats.

    Returns:
        Dict with path, species, valid, width, height, size_bytes, sha256.
    """

    filename = os.path.basename(filepath)
    species = os.path.basename(os.path.dirname(filepath))

    result = {
        "path": filepath,
        "filename": filename,
        "species": species,
        "valid": False,
        "width": 0,
        "height": 0,
        "size_bytes": 0,
        "sha256": "",
        "error": "",
    }

    try:
        # File size
        result["size_bytes"] = os.path.getsize(filepath)

        # PIL check
        pil_img = Image.open(filepath)
        pil_img.verify()

        # Re-open after verify (verify consumes the image)
        pil_img = Image.open(filepath)
        w, h = pil_img.size
        result["width"] = w
        result["height"] = h

        # OpenCV check
        cv_img = cv2.imread(filepath)
        if cv_img is None:
            result["error"] = "opencv_unreadable"
            return result

        # SHA-256 for cross-species dedup
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha.update(chunk)
        result["sha256"] = sha.hexdigest()

        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# COLLECT ALL IMAGES FROM A DIRECTORY
# ============================================================

def collect_images(base_dir: str) -> Dict[str, List[str]]:
    """
    Collect all image paths organized by species.

    Args:
        base_dir: Base directory containing species subdirectories.

    Returns:
        Dict mapping species_key → list of image paths.
    """

    images: Dict[str, List[str]] = {}

    for species_key in SPECIES_LIST:
        species_dir = os.path.join(base_dir, species_key)
        paths = []

        if os.path.exists(species_dir):
            for f in sorted(os.listdir(species_dir)):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(species_dir, f))

        images[species_key] = paths

    return images


# ============================================================
# VERIFY ALL IMAGES
# ============================================================

def verify_dataset(
    base_dir: str,
    source_name: str = "cleaned",
) -> Dict:
    """
    Run full verification on the dataset.

    Args:
        base_dir: Directory containing species subdirectories.
        source_name: Name for logging ('cleaned' or 'species').

    Returns:
        Dict with verification results.
    """

    print(f"\n{'=' * 60}")
    print(f"  TreeVisionAI — Dataset Verification")
    print(f"  Source: {base_dir}")
    print(f"{'=' * 60}\n")

    # Collect all images
    species_images = collect_images(base_dir)

    # Count total
    total = sum(len(v) for v in species_images.values())

    if total == 0:
        logger.warning(f"No images found in {base_dir}. Nothing to verify.")
        return {}

    logger.info(f"Found {total} images across {len(species_images)} species.\n")

    # Scan all images
    all_paths = []
    for paths in species_images.values():
        all_paths.extend(paths)

    num_workers = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        all_results = list(
            tqdm(
                executor.map(scan_image, all_paths),
                total=len(all_paths),
                desc="  Verifying",
                unit="img",
            )
        )

    # --------------------------------
    # Organize results by species
    # --------------------------------
    species_stats: Dict[str, Dict] = {}
    all_hashes: Dict[str, List[str]] = defaultdict(list)  # hash → [species]
    invalid_files: List[Dict] = []

    for species_key in SPECIES_LIST:

        species_results = [
            r for r in all_results if r["species"] == species_key
        ]

        valid = [r for r in species_results if r["valid"]]
        invalid = [r for r in species_results if not r["valid"]]

        if valid:
            widths = [r["width"] for r in valid]
            heights = [r["height"] for r in valid]
            sizes = [r["size_bytes"] for r in valid]

            stats = {
                "total": len(species_results),
                "valid": len(valid),
                "invalid": len(invalid),
                "min_width": min(widths),
                "max_width": max(widths),
                "avg_width": sum(widths) // len(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "avg_height": sum(heights) // len(heights),
                "total_size_mb": round(sum(sizes) / (1024 * 1024), 1),
                "avg_size_kb": round(
                    sum(sizes) / len(sizes) / 1024, 1
                ),
            }
        else:
            stats = {
                "total": len(species_results),
                "valid": 0,
                "invalid": len(invalid),
            }

        species_stats[species_key] = stats
        invalid_files.extend(invalid)

        # Track hashes for cross-species dedup
        for r in valid:
            all_hashes[r["sha256"]].append(species_key)

    # --------------------------------
    # Cross-species duplicate check
    # --------------------------------
    cross_duplicates = {
        h: species_list
        for h, species_list in all_hashes.items()
        if len(set(species_list)) > 1
    }

    # --------------------------------
    # Print report
    # --------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Verification Results")
    print(f"{'=' * 60}\n")

    # Per-species table
    print(f"  {'Species':<12} {'Total':>6} {'Valid':>6} "
          f"{'Invalid':>8} {'Avg Res':>12} {'Size (MB)':>10}")
    print(f"  {'-' * 56}")

    for key in SPECIES_LIST:
        s = species_stats[key]
        name = SPECIES[key]["display_name"]
        total_count = s["total"]
        valid_count = s["valid"]
        invalid_count = s["invalid"]

        if valid_count > 0:
            avg_res = f"{s['avg_width']}x{s['avg_height']}"
            size_mb = f"{s['total_size_mb']}"
        else:
            avg_res = "N/A"
            size_mb = "0"

        print(
            f"  {name:<12} {total_count:>6} {valid_count:>6} "
            f"{invalid_count:>8} {avg_res:>12} {size_mb:>10}"
        )

    # Class balance check
    valid_counts = [
        species_stats[k]["valid"] for k in SPECIES_LIST
    ]
    if valid_counts and max(valid_counts) > 0:
        min_count = min(valid_counts)
        max_count = max(valid_counts)
        ratio = min_count / max_count if max_count > 0 else 0

        print(f"\n  Class Balance Ratio: {ratio:.2f}")
        print()

        if ratio < 0.5:
            logger.warning(
                "[WARNING] Significant class imbalance detected! "
                f"Min={min_count}, Max={max_count}. "
                "Consider collecting more images for underrepresented species."
            )
        elif ratio < 0.75:
            logger.info(
                "  Moderate class imbalance. "
                "WeightedRandomSampler will handle this during training."
            )
        else:
            logger.info("  [OK] Classes are reasonably balanced.")

    # Cross-species duplicates
    if cross_duplicates:
        print(f"\n  [WARNING] Cross-species duplicates: {len(cross_duplicates)}")
        for h, species in list(cross_duplicates.items())[:5]:
            print(f"    Hash {h[:12]}... found in: {species}")
    else:
        print(f"\n  [OK] No cross-species duplicates found.")

    # Invalid files
    if invalid_files:
        print(f"\n  [WARNING] Invalid files: {len(invalid_files)}")
        for f in invalid_files[:10]:
            print(f"    {f['filename']} ({f['species']}): {f['error']}")

    print(f"\n{'=' * 60}\n")

    # --------------------------------
    # Save report CSV
    # --------------------------------
    report_path = os.path.join(REPORTS_DIR, "verification_report.csv")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "species", "filename", "valid", "width", "height",
            "size_bytes", "error",
        ])

        for r in all_results:
            writer.writerow([
                r["species"], r["filename"], r["valid"],
                r["width"], r["height"], r["size_bytes"], r["error"],
            ])

    logger.info(f"Report saved: {report_path}")

    return {
        "species_stats": species_stats,
        "cross_duplicates": len(cross_duplicates),
        "invalid_files": len(invalid_files),
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Verify dataset integrity"
    )

    parser.add_argument(
        "--source",
        choices=["cleaned", "species"],
        default="cleaned",
        help="Which dataset to verify: 'cleaned' or 'species' (split). Default: cleaned",
    )

    args = parser.parse_args()

    if args.source == "cleaned":
        verify_dataset(CLEANED_DIR, source_name="cleaned")
    else:
        # For split dataset, verify train/val/test separately
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(SPECIES_DIR, split)
            if os.path.exists(split_dir):
                verify_dataset(split_dir, source_name=f"species/{split}")
