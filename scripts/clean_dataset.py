"""
TreeVisionAI — Dataset Cleaner
===============================

Production-quality image cleaning pipeline.
Processes raw images per species and filters out:
  - Corrupted images
  - Too small images (< MIN_WIDTH x MIN_HEIGHT)
  - Blurry images (Laplacian variance < threshold)
  - Too dark images (mean brightness < threshold)
  - Overexposed images (mean brightness > threshold)
  - Exact duplicates (SHA-256 hash)
  - Near duplicates (perceptual hash)

Accepted images → datasets/cleaned/<species>/
Uncertain images → datasets/review/<species>/
Generates a cleaning report per species.

Usage:
    python scripts/clean_dataset.py
    python scripts/clean_dataset.py --species raintree mango ashoka
"""

import os
import sys
import csv
import hashlib
import argparse
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
from PIL import Image
import imagehash

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES,
    SPECIES_LIST,
    RAW_DIR,
    CLEANED_DIR,
    REVIEW_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    IMAGE_EXTENSIONS,
    MIN_WIDTH,
    MIN_HEIGHT,
    BLUR_THRESHOLD,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
    PHASH_THRESHOLD,
    setup_logging,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="cleaner",
    log_file=os.path.join(LOGS_DIR, "clean_dataset.log"),
)


# ============================================================
# IMAGE QUALITY CHECKS
# ============================================================

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for exact duplicate detection."""

    h = hashlib.sha256()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def compute_phash(filepath: str) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for near-duplicate detection."""

    try:
        img = Image.open(filepath).convert("RGB")
        return imagehash.phash(img)
    except Exception:
        return None


def compute_blur_score(image: any) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher value = sharper image. Lower value = blurrier.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness(image: any) -> float:
    """
    Compute mean brightness (0-255).
    Low = too dark, High = overexposed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def check_image_quality(filepath: str) -> Dict[str, any]:
    """
    Run all quality checks on a single image.

    Returns:
        Dict with keys: path, filename, status, reason, width, height,
        blur_score, brightness, sha256, phash
    """

    filename = os.path.basename(filepath)
    result = {
        "path": filepath,
        "filename": filename,
        "status": "accepted",
        "reason": "",
        "width": 0,
        "height": 0,
        "blur_score": 0.0,
        "brightness": 0.0,
        "sha256": "",
        "phash": None,
    }

    # ----------------------------------
    # Check 1: Can we read the file?
    # ----------------------------------
    try:
        img = cv2.imread(filepath)

        if img is None:
            result["status"] = "rejected"
            result["reason"] = "corrupted"
            return result

    except Exception:
        result["status"] = "rejected"
        result["reason"] = "corrupted"
        return result

    # ----------------------------------
    # Check 2: Image dimensions
    # ----------------------------------
    h, w = img.shape[:2]
    result["width"] = w
    result["height"] = h

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        result["status"] = "rejected"
        result["reason"] = "too_small"
        return result

    # ----------------------------------
    # Check 3: Blur detection
    # ----------------------------------
    blur = compute_blur_score(img)
    result["blur_score"] = round(blur, 2)

    if blur < BLUR_THRESHOLD:
        result["status"] = "review"
        result["reason"] = "blurry"
        return result

    # ----------------------------------
    # Check 4: Brightness / exposure
    # ----------------------------------
    brightness = compute_brightness(img)
    result["brightness"] = round(brightness, 2)

    if brightness < BRIGHTNESS_MIN:
        result["status"] = "review"
        result["reason"] = "too_dark"
        return result

    if brightness > BRIGHTNESS_MAX:
        result["status"] = "review"
        result["reason"] = "overexposed"
        return result

    # ----------------------------------
    # Check 5: Compute hashes for dedup
    # ----------------------------------
    result["sha256"] = compute_sha256(filepath)
    result["phash"] = compute_phash(filepath)

    return result


# ============================================================
# SCAN ALL IMAGES IN PARALLEL
# ============================================================

def scan_images(image_paths: List[str]) -> List[Dict[str, any]]:
    """
    Scan all images in parallel for quality issues.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of quality check results.
    """

    results = []
    num_workers = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(check_image_quality, image_paths),
                total=len(image_paths),
                desc="  Scanning",
                unit="img",
            )
        )

    return results


# ============================================================
# DEDUPLICATE
# ============================================================

def deduplicate(
    results: List[Dict[str, any]],
) -> List[Dict[str, any]]:
    """
    Mark exact and near-duplicate images.

    Processes only images that passed quality checks (status=accepted).
    Marks duplicates with status='rejected', reason='exact_duplicate'
    or 'near_duplicate'.

    Args:
        results: List of quality check results.

    Returns:
        Updated results with duplicates marked.
    """

    seen_sha: Set[str] = set()
    seen_phash: List[imagehash.ImageHash] = []

    exact_count = 0
    near_count = 0

    for r in results:

        if r["status"] != "accepted":
            continue

        sha = r["sha256"]
        phash = r["phash"]

        # Exact duplicate check
        if sha in seen_sha:
            r["status"] = "rejected"
            r["reason"] = "exact_duplicate"
            exact_count += 1
            continue

        seen_sha.add(sha)

        # Near-duplicate check (perceptual hash)
        if phash is not None:
            is_near_dup = False

            for saved_phash in seen_phash:
                if phash - saved_phash <= PHASH_THRESHOLD:
                    r["status"] = "rejected"
                    r["reason"] = "near_duplicate"
                    near_count += 1
                    is_near_dup = True
                    break

            if not is_near_dup:
                seen_phash.append(phash)

    logger.info(
        f"  Duplicates found: {exact_count} exact, {near_count} near-duplicate"
    )

    return results


# ============================================================
# ORGANIZE FILES (COPY TO CLEANED / REVIEW)
# ============================================================

def organize_files(
    results: List[Dict[str, any]],
    species_key: str,
) -> Dict[str, int]:
    """
    Copy accepted images to cleaned folder, uncertain to review folder.

    Args:
        results: Quality check results.
        species_key: Species folder key.

    Returns:
        Dict with counts per status.
    """

    import shutil

    cleaned_dir = os.path.join(CLEANED_DIR, species_key)
    review_dir = os.path.join(REVIEW_DIR, species_key)

    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    counts = {
        "accepted": 0,
        "review": 0,
        "rejected_corrupted": 0,
        "rejected_too_small": 0,
        "rejected_exact_duplicate": 0,
        "rejected_near_duplicate": 0,
        "rejected_blurry": 0,
        "rejected_too_dark": 0,
        "rejected_overexposed": 0,
    }

    for r in tqdm(results, desc="  Organizing", unit="img"):

        src = r["path"]
        filename = r["filename"]
        status = r["status"]
        reason = r["reason"]

        if status == "accepted":
            dst = os.path.join(cleaned_dir, filename)
            counts["accepted"] += 1

        elif status == "review":
            dst = os.path.join(review_dir, filename)
            key = f"review"
            counts["review"] += 1

        else:
            # Rejected — don't copy, just count
            key = f"rejected_{reason}"
            if key in counts:
                counts[key] += 1
            continue

        # Copy (not move) to preserve raw data
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.warning(f"Failed to copy {filename}: {e}")

    return counts


# ============================================================
# GENERATE CLEANING REPORT
# ============================================================

def generate_report(
    results: List[Dict[str, any]],
    species_key: str,
) -> str:
    """
    Save a CSV report of all images and their quality status.

    Returns:
        Path to the saved report.
    """

    report_path = os.path.join(
        REPORTS_DIR, f"{species_key}_cleaning_report.csv"
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "status", "reason",
            "width", "height", "blur_score", "brightness",
        ])

        for r in results:
            writer.writerow([
                r["filename"],
                r["status"],
                r["reason"],
                r["width"],
                r["height"],
                r["blur_score"],
                r["brightness"],
            ])

    return report_path


# ============================================================
# CLEAN A SINGLE SPECIES
# ============================================================

def clean_species(species_key: str) -> Dict[str, int]:
    """
    Full cleaning pipeline for one species.

    1. Scan all raw images for quality
    2. Deduplicate
    3. Copy accepted → cleaned, uncertain → review
    4. Generate report

    Args:
        species_key: Species folder key.

    Returns:
        Dict with counts per status.
    """

    species_info = SPECIES[species_key]
    display_name = species_info["display_name"]
    raw_dir = os.path.join(RAW_DIR, species_key)

    logger.info(
        f"\n{'=' * 50}\n"
        f"  Cleaning: {display_name}\n"
        f"  Source: {raw_dir}\n"
        f"{'=' * 50}"
    )

    # Collect all image paths
    image_paths = []
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(raw_dir, f))

    if not image_paths:
        logger.warning(f"No images found in {raw_dir}. Skipping.")
        return {"accepted": 0}

    logger.info(f"  Found {len(image_paths)} raw images.")

    # Step 1: Scan quality
    results = scan_images(image_paths)

    # Step 2: Deduplicate
    results = deduplicate(results)

    # Step 3: Organize files
    counts = organize_files(results, species_key)

    # Step 4: Generate report
    report_path = generate_report(results, species_key)

    # Print summary
    logger.info(
        f"\n  {display_name} — Cleaning Summary\n"
        f"  {'─' * 40}\n"
        f"  Accepted           : {counts['accepted']}\n"
        f"  Sent to review     : {counts['review']}\n"
        f"  Rejected (corrupt) : {counts['rejected_corrupted']}\n"
        f"  Rejected (small)   : {counts['rejected_too_small']}\n"
        f"  Rejected (exact dup) : {counts['rejected_exact_duplicate']}\n"
        f"  Rejected (near dup)  : {counts['rejected_near_duplicate']}\n"
        f"  Report saved       : {report_path}\n"
    )

    return counts


# ============================================================
# MAIN
# ============================================================

def main(species_keys: Optional[List[str]] = None) -> None:
    """
    Clean raw images for all or selected species.

    Args:
        species_keys: List of species to clean. None = all.
    """

    if species_keys is None:
        species_keys = SPECIES_LIST

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Dataset Cleaner")
    print("=" * 60)
    print(f"\n  Species to clean: {species_keys}\n")

    all_counts = {}

    for key in species_keys:

        if key not in SPECIES:
            logger.error(f"Unknown species: '{key}'")
            continue

        counts = clean_species(key)
        all_counts[key] = counts

    # Final summary
    print("\n" + "=" * 60)
    print("  Cleaning Summary — All Species")
    print("=" * 60)

    for key, counts in all_counts.items():
        name = SPECIES[key]["display_name"]
        accepted = counts.get("accepted", 0)
        review = counts.get("review", 0)
        total_rejected = sum(
            v for k, v in counts.items()
            if k.startswith("rejected")
        )
        print(
            f"  {name:<15} "
            f"Accepted: {accepted:>5}  "
            f"Review: {review:>5}  "
            f"Rejected: {total_rejected:>5}"
        )

    print("\n" + "=" * 60 + "\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Clean raw dataset images"
    )

    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        choices=SPECIES_LIST,
        help=f"Species to clean. Default: all ({SPECIES_LIST})",
    )

    args = parser.parse_args()

    main(species_keys=args.species)