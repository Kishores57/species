"""
TreeVisionAI — Species Image Downloader
========================================

Downloads high-resolution leaf images from iNaturalist
for all configured species. Supports resuming interrupted
downloads and saves metadata per species.

Usage:
    python scripts/download_species.py
    python scripts/download_species.py --species raintree mango ashoka
    python scripts/download_species.py --species mango --target 3000
"""

import os
import sys
import csv
import time
import argparse
import requests
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    SPECIES,
    SPECIES_LIST,
    RAW_DIR,
    LOGS_DIR,
    DOWNLOAD_TARGET_PER_SPECIES,
    DOWNLOAD_THREADS,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_PER_PAGE,
    INAT_API_URL,
    INAT_USER_AGENT,
    IMAGE_EXTENSIONS,
    setup_logging,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logging(
    name="downloader",
    log_file=os.path.join(LOGS_DIR, "download.log"),
)


# ============================================================
# HELPER: GET EXISTING IMAGE COUNT
# ============================================================

def get_existing_images(folder: str) -> Set[str]:
    """
    Get set of existing image filenames in a folder.
    Used for resume support — skips already downloaded files.
    """
    existing = set()

    if not os.path.exists(folder):
        return existing

    for f in os.listdir(folder):
        if f.lower().endswith(IMAGE_EXTENSIONS):
            existing.add(f)

    return existing


# ============================================================
# HELPER: LOAD EXISTING METADATA
# ============================================================

def load_existing_metadata(csv_path: str) -> List[List[str]]:
    """Load existing metadata rows from CSV if it exists."""

    rows = []

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                rows = [row for row in reader]
        except Exception:
            pass

    return rows


# ============================================================
# COLLECT IMAGE URLS FROM iNATURALIST
# ============================================================

def _fetch_page(
    session: requests.Session,
    scientific_name: str,
    page: int,
    extra_params: Dict = None,
) -> List[Dict]:
    """
    Fetch a single page of observations from iNaturalist.

    Args:
        session: Requests session with User-Agent set.
        scientific_name: Species scientific name.
        page: Page number (1-indexed).
        extra_params: Additional query params to merge.

    Returns:
        List of observation dicts from the API response.
    """

    params = {
        "taxon_name": scientific_name,
        "photos": "true",
        "per_page": DOWNLOAD_PER_PAGE,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }

    if extra_params:
        params.update(extra_params)

    response = session.get(
        INAT_API_URL,
        params=params,
        timeout=DOWNLOAD_TIMEOUT,
    )

    if response.status_code == 429:
        logger.warning("Rate limited by iNaturalist. Waiting 60s...")
        time.sleep(60)
        # Retry the same page
        return _fetch_page(session, scientific_name, page, extra_params)

    if response.status_code != 200:
        logger.error(f"API error: {response.status_code}")
        return []

    return response.json().get("results", [])


def _extract_images_from_results(
    results: List[Dict],
    existing_filenames: Set[str],
    seen_photo_ids: Set[str],
) -> List[Dict[str, str]]:
    """
    Extract image info dicts from API observation results.

    Skips already-downloaded files and photos seen in earlier passes.

    Args:
        results: Observation dicts from iNaturalist API.
        existing_filenames: Filenames already on disk.
        seen_photo_ids: Photo IDs already collected (avoids cross-pass dupes).

    Returns:
        List of image info dicts.
    """

    images: List[Dict[str, str]] = []

    for obs in results:
        for photo in obs.get("photos", []):
            photo_id = str(photo.get("id", ""))
            filename = f"{photo_id}.jpg"

            # Skip already downloaded or already collected
            if filename in existing_filenames or photo_id in seen_photo_ids:
                continue

            # Get highest resolution URL
            url = photo.get("original_url")
            if not url:
                url = photo.get("url", "").replace("square", "original")

            if not url:
                continue

            seen_photo_ids.add(photo_id)

            images.append({
                "url": url,
                "filename": filename,
                "observation_id": str(obs.get("id", "")),
                "date": obs.get("observed_on", ""),
                "license": photo.get("license_code", ""),
            })

    return images


def collect_image_urls(
    scientific_name: str,
    target_count: int,
    existing_filenames: Set[str],
) -> List[Dict[str, str]]:
    """
    Query iNaturalist API to collect image URLs for a species.

    Uses a multi-pass fallback strategy to maximize image yield:
      Pass 1: Research-grade + leaf annotations (highest quality)
      Pass 2: Research-grade, any plant part (broader)
      Pass 3: Any quality grade with photos (broadest)

    Skips images that are already downloaded (by filename).

    Args:
        scientific_name: Species scientific name for API query.
        target_count: Number of new images to collect.
        existing_filenames: Set of already-downloaded filenames.

    Returns:
        List of dicts with url, filename, observation_id, date, license.
    """

    session = requests.Session()
    session.headers.update({"User-Agent": INAT_USER_AGENT})

    # Define search passes: most specific → broadest
    search_passes = [
        {
            "label": "Pass 1 — Research-grade, leaf-annotated",
            "params": {
                "quality_grade": "research",
                "term_id": 12,        # Plant Phenology
                "term_value_id": 13,  # Leaves
            },
        },
        {
            "label": "Pass 2 — Research-grade, all photos",
            "params": {
                "quality_grade": "research",
            },
        },
        {
            "label": "Pass 3 — Any quality, all photos",
            "params": {},
        },
    ]

    all_images: List[Dict[str, str]] = []
    seen_photo_ids: Set[str] = set()

    logger.info(
        f"Collecting URLs for '{scientific_name}' "
        f"(target: {target_count}, existing: {len(existing_filenames)})"
    )

    for pass_info in search_passes:

        if len(all_images) >= target_count:
            break

        label = pass_info["label"]
        extra_params = pass_info["params"]
        remaining = target_count - len(all_images)

        logger.info(f"  {label} (need {remaining} more)...")

        page = 1
        empty_pages = 0

        while len(all_images) < target_count:

            try:
                results = _fetch_page(
                    session, scientific_name, page, extra_params
                )

                if not results:
                    empty_pages += 1
                    if empty_pages >= 3:
                        logger.info(
                            f"    No more observations in this pass "
                            f"(collected {len(all_images)} so far)."
                        )
                        break
                    page += 1
                    continue

                empty_pages = 0

                new_images = _extract_images_from_results(
                    results, existing_filenames, seen_photo_ids
                )

                # Add images up to target
                for img in new_images:
                    if len(all_images) >= target_count:
                        break
                    all_images.append(img)

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on page {page}. Retrying...")
                time.sleep(5)
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                time.sleep(10)
                continue

            page += 1

            # Respectful delay between requests
            time.sleep(1)

        logger.info(
            f"    {label} complete. Total collected: {len(all_images)}"
        )

    logger.info(f"Collected {len(all_images)} new image URLs total.")
    return all_images


# ============================================================
# DOWNLOAD A SINGLE IMAGE
# ============================================================

def download_single_image(
    url: str,
    filepath: str,
    max_retries: int = DOWNLOAD_MAX_RETRIES,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> bool:
    """
    Download a single image with retry logic.

    Args:
        url: Image URL to download.
        filepath: Local path to save the image.
        max_retries: Number of retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        True if download succeeded, False otherwise.
    """

    for attempt in range(max_retries):

        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": INAT_USER_AGENT},
            )

            if response.status_code == 200:
                # Verify it's actually image data (at least 1KB)
                if len(response.content) < 1024:
                    return False

                with open(filepath, "wb") as f:
                    f.write(response.content)

                return True

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    return False


# ============================================================
# DOWNLOAD ALL IMAGES FOR A SPECIES
# ============================================================

def download_species(
    species_key: str,
    target_count: int = DOWNLOAD_TARGET_PER_SPECIES,
) -> Dict[str, int]:
    """
    Download images for a single species from iNaturalist.

    Args:
        species_key: Species folder key (e.g., 'raintree').
        target_count: Target number of images to download.

    Returns:
        Dict with download statistics.
    """

    species_info = SPECIES[species_key]
    scientific_name = species_info["scientific_name"]
    display_name = species_info["display_name"]

    output_dir = os.path.join(RAW_DIR, species_key)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "metadata.csv")

    logger.info(
        f"\n{'=' * 50}\n"
        f"  Downloading: {display_name} ({scientific_name})\n"
        f"  Target: {target_count} images\n"
        f"  Output: {output_dir}\n"
        f"{'=' * 50}"
    )

    # Check existing images for resume support
    existing = get_existing_images(output_dir)
    needed = max(0, target_count - len(existing))

    if needed == 0:
        logger.info(
            f"Already have {len(existing)} images for {display_name}. "
            f"Skipping download."
        )
        return {"species": species_key, "existing": len(existing),
                "downloaded": 0, "failed": 0}

    logger.info(
        f"Found {len(existing)} existing images. "
        f"Need {needed} more."
    )

    # Collect URLs
    image_list = collect_image_urls(scientific_name, needed, existing)

    if not image_list:
        logger.warning(f"No new images found for {display_name}.")
        return {"species": species_key, "existing": len(existing),
                "downloaded": 0, "failed": 0}

    # Download images with thread pool
    downloaded = 0
    failed = 0
    metadata_rows = load_existing_metadata(csv_path)

    pbar = tqdm(
        total=len(image_list),
        desc=f"  {display_name}",
        unit="img",
    )

    def _download_task(img_info: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Download task for thread pool."""
        filepath = os.path.join(output_dir, img_info["filename"])
        success = download_single_image(img_info["url"], filepath)
        return img_info if success else None

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:

        futures = {
            executor.submit(_download_task, img): img
            for img in image_list
        }

        for future in as_completed(futures):

            result = future.result()

            if result is not None:
                downloaded += 1
                metadata_rows.append([
                    result["filename"],
                    result["observation_id"],
                    result["date"],
                    result["license"],
                    result["url"],
                ])
            else:
                failed += 1

            pbar.update(1)

    pbar.close()

    # Save metadata CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "observation_id",
            "observed_on",
            "license",
            "image_url",
        ])
        writer.writerows(metadata_rows)

    logger.info(
        f"\n  {display_name} — Done!\n"
        f"  Downloaded : {downloaded}\n"
        f"  Failed     : {failed}\n"
        f"  Total      : {len(existing) + downloaded}\n"
    )

    return {
        "species": species_key,
        "existing": len(existing),
        "downloaded": downloaded,
        "failed": failed,
    }


# ============================================================
# MAIN — DOWNLOAD ALL SPECIES
# ============================================================

def main(
    species_keys: Optional[List[str]] = None,
    target: int = DOWNLOAD_TARGET_PER_SPECIES,
) -> None:
    """
    Download images for all (or selected) species.

    Args:
        species_keys: List of species to download. None = all species.
        target: Target image count per species.
    """

    if species_keys is None:
        species_keys = SPECIES_LIST

    # Validate species keys
    for key in species_keys:
        if key not in SPECIES:
            logger.error(
                f"Unknown species: '{key}'. "
                f"Valid options: {SPECIES_LIST}"
            )
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  TreeVisionAI — Dataset Downloader")
    print("=" * 60)
    print(f"\n  Species to download: {species_keys}")
    print(f"  Target per species : {target}")
    print(f"  Download threads   : {DOWNLOAD_THREADS}")
    print(f"  Source             : iNaturalist (research grade)")
    print()

    all_stats = []

    for key in species_keys:
        stats = download_species(key, target)
        all_stats.append(stats)

    # Print summary
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    print(f"\n  {'Species':<15} {'Existing':>10} {'Downloaded':>12} {'Failed':>8} {'Total':>8}")
    print("  " + "-" * 53)

    for s in all_stats:
        total = s["existing"] + s["downloaded"]
        name = SPECIES[s["species"]]["display_name"]
        print(
            f"  {name:<15} {s['existing']:>10} "
            f"{s['downloaded']:>12} {s['failed']:>8} {total:>8}"
        )

    print("\n" + "=" * 60 + "\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TreeVisionAI — Download species images from iNaturalist"
    )

    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        choices=SPECIES_LIST,
        help=f"Species to download. Default: all ({SPECIES_LIST})",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=DOWNLOAD_TARGET_PER_SPECIES,
        help=f"Target images per species (default: {DOWNLOAD_TARGET_PER_SPECIES})",
    )

    args = parser.parse_args()

    main(
        species_keys=args.species,
        target=args.target,
    )
