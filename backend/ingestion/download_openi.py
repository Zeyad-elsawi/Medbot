"""
backend/ingestion/download_openi.py
Downloads the OpenI chest X-ray dataset (Indiana University).
3,996 frontal X-ray images + paired radiology reports — no registration required.
Run once locally. Output goes to data/openi/.
"""

import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

# Add parent dir to path so we can import core config
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# OpenI dataset — publicly available, no registration needed
OPENI_REPORT_URL = "https://openi.nlm.nih.gov/api/search?m=1&n=100&query=chest&coll=iu&sp=1&se=40"
OPENI_REPORTS_XML_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
OPENI_PNG_BASE_URL = "https://openi.nlm.nih.gov/imgs/OAI/"


def create_output_dirs(settings) -> tuple[Path, Path]:
    """
    Create output directories for images and reports.

    Returns:
        Tuple of (images_dir, reports_dir) as Path objects.
    """
    base = Path(settings.openi_data_dir)
    images_dir = base / "images"
    reports_dir = base / "reports"
    images_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories: %s, %s", images_dir, reports_dir)
    return images_dir, reports_dir


def fetch_openi_records(max_records: int = 4000) -> list[dict]:
    """
    Fetch all OpenI record metadata from the NLM search API.

    Uses pagination (100 records per page) to retrieve all available records.
    Each record contains image filename and report text fields.

    Args:
        max_records: Maximum number of records to fetch (default: 4000 ≈ full dataset)

    Returns:
        List of record dicts with 'image_url', 'report', 'uid', 'findings', 'impression' keys.

    Raises:
        requests.HTTPError: If any paginated request fails.
    """
    records = []
    page_size = 100
    page = 1
    session = requests.Session()
    session.headers.update({"User-Agent": "MedBot-Research/1.0 (portfolio project)"})

    logger.info("Fetching OpenI metadata (up to %d records)...", max_records)

    while len(records) < max_records:
        start = (page - 1) * page_size + 1
        url = (
            f"https://openi.nlm.nih.gov/api/search"
            f"?m={start}&n={page_size}&query=chest&coll=iu&sp=1&se=3"
        )

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            logger.error("HTTP error fetching page %d: %s", page, exc)
            raise
        except requests.RequestException as exc:
            logger.error("Network error fetching page %d: %s", page, exc)
            raise

        results = data.get("list", [])
        if not results:
            logger.info("No more records at page %d — stopping.", page)
            break

        for item in results:
            img_url = item.get("imgLarge", "") or item.get("imgSmall", "")
            if not img_url:
                continue
            filename = img_url.split("/")[-1]
            records.append(
                {
                    "uid": item.get("uid", ""),
                    "image_filename": filename,
                    "image_url": img_url,
                    "findings": item.get("MeSH", {}).get("major", []),
                    "report_sections": item.get("abstract", {}),
                    "diagnosis_label": ", ".join(
                        item.get("MeSH", {}).get("major", ["unknown"])
                    ),
                }
            )

        logger.info("Page %d: fetched %d records (total so far: %d)", page, len(results), len(records))
        page += 1

        if len(results) < page_size:
            break  # Last page

    logger.info("Total records fetched: %d", len(records))
    return records


def download_image(
    session: requests.Session, image_url: str, dest: Path
) -> bool:
    """
    Download a single chest X-ray image from OpenI.

    Args:
        session: requests.Session with headers already set.
        image_url: Full path suffix from the API (e.g. '/imgs/512/230/3037/CXR3037_IM-1410-1001.png').
        dest: Destination Path to save the image.

    Returns:
        True if download succeeded, False if skipped (already exists) or failed.
    """
    if dest.exists():
        return False  # Skip — already downloaded

    url = f"https://openi.nlm.nih.gov{image_url}"
    try:
        resp = session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as exc:
        logger.warning("Failed to download %s: %s", filename, exc)
        return False


def save_metadata(records: list[dict], reports_dir: Path) -> None:
    """
    Save all record metadata as a JSON file for ingestion pipeline.

    Args:
        records: List of record dicts from fetch_openi_records.
        reports_dir: Directory to save metadata.json.
    """
    metadata_path = reports_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s", metadata_path)


def main() -> None:
    """
    Main download orchestrator.
    Fetches metadata, downloads images, saves reports to disk.
    """
    settings = get_settings()
    images_dir, reports_dir = create_output_dirs(settings)

    # Fetch record metadata
    try:
        records = fetch_openi_records(max_records=4000)
    except Exception as exc:
        logger.error("Failed to fetch OpenI records: %s", exc)
        sys.exit(1)

    if not records:
        logger.error("No records returned from OpenI API. Check your network connection.")
        sys.exit(1)

    # Save metadata first (so we have it even if image download partially fails)
    save_metadata(records, reports_dir)

    # Download images
    session = requests.Session()
    session.headers.update({"User-Agent": "MedBot-Research/1.0 (portfolio project)"})

    downloaded = 0
    skipped = 0
    failed = 0

    logger.info("Downloading %d images to %s ...", len(records), images_dir)

    for record in tqdm(records, desc="Downloading X-rays", unit="img"):
        filename = record.get("image_filename", "")
        img_url = record.get("image_url", "")
        if not filename or not img_url:
            failed += 1
            continue

        dest = images_dir / filename
        result = download_image(session, img_url, dest)

        if result is True:
            downloaded += 1
        elif result is False and dest.exists():
            skipped += 1
        else:
            failed += 1

    logger.info(
        "Download complete — Downloaded: %d | Skipped (already exists): %d | Failed: %d",
        downloaded,
        skipped,
        failed,
    )
    logger.info("Images saved to: %s", images_dir)
    logger.info("Metadata saved to: %s", reports_dir / "metadata.json")


if __name__ == "__main__":
    main()
