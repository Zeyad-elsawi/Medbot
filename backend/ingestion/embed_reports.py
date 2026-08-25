"""
backend/ingestion/embed_reports.py
Generates PubMedBERT embeddings for all OpenI radiology report texts.
Saves embeddings + metadata for the clinical_notes Qdrant collection.
Run after download_openi.py completes.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # PubMedBERT / S-PubMedBert-MS-MARCO output dimension


def load_pubmedbert(model_name: str) -> SentenceTransformer:
    """
    Load S-PubMedBert-MS-MARCO sentence transformer model.

    Model is downloaded from HuggingFace on first run and cached locally.
    Subsequent runs use the cached version — no download needed.

    Args:
        model_name: HuggingFace model ID (from settings).

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        RuntimeError: If model download or load fails.
    """
    try:
        logger.info("Loading PubMedBERT model: %s ...", model_name)
        model = SentenceTransformer(model_name)
        logger.info("PubMedBERT loaded. Embedding dim: %d", model.get_sentence_embedding_dimension())
        return model
    except Exception as exc:
        logger.error("Failed to load PubMedBERT: %s", exc)
        raise RuntimeError(f"PubMedBERT load failed: {exc}") from exc


def extract_report_text(record: dict) -> str:
    """
    Extract and combine all text sections from an OpenI metadata record.

    OpenI records have findings and impression sections in 'report_sections'.
    We combine them into a single string for embedding.

    Args:
        record: Single record dict from metadata.json.

    Returns:
        Combined report text string. Returns empty string if no text found.
    """
    sections = record.get("report_sections", {})
    if not sections:
        # Fallback: use diagnosis label + findings list
        label = record.get("diagnosis_label", "")
        findings = " ".join(record.get("findings", []))
        return f"{label}. {findings}".strip()

    if isinstance(sections, dict):
        # Extract all values from the sections dict
        parts = []
        for key in ["findings", "impression", "comparison", "indication"]:
            val = sections.get(key, "")
            if val:
                parts.append(f"{key.capitalize()}: {val}")
        return " ".join(parts)

    if isinstance(sections, str):
        return sections

    return ""


def main() -> None:
    """
    Main embedding pipeline for OpenI radiology reports.

    Reads metadata from data/openi/reports/metadata.json,
    extracts report texts, embeds them with PubMedBERT,
    saves to data/openi/report_embeddings.npy and report_metadata.json.
    """
    settings = get_settings()

    metadata_path = Path(settings.openi_data_dir) / "reports" / "metadata.json"

    if not metadata_path.exists():
        logger.error("Metadata not found: %s — run download_openi.py first", metadata_path)
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Loaded %d records from metadata.", len(records))

    # Extract report texts
    texts = []
    valid_records = []
    for rec in records:
        text = extract_report_text(rec)
        if text and len(text.strip()) > 20:  # Skip empty/trivial texts
            texts.append(text)
            valid_records.append(rec)
        else:
            logger.debug("Skipping record %s — empty report text", rec.get("uid", "?"))

    logger.info("Valid reports with text: %d / %d", len(texts), len(records))

    if not texts:
        logger.error("No valid report texts found. Check metadata format.")
        sys.exit(1)

    # Load model
    model = load_pubmedbert(settings.pubmedbert_model)

    # Embed all texts in batches
    logger.info("Generating embeddings for %d reports...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=settings.openi_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        convert_to_numpy=True,
    )

    logger.info("Embeddings shape: %s", embeddings.shape)

    # Save outputs
    output_dir = Path(settings.openi_data_dir)
    np.save(output_dir / "report_embeddings.npy", embeddings.astype(np.float32))

    report_metadata = [
        {
            "uid": rec["uid"],
            "diagnosis_label": rec.get("diagnosis_label", "unknown"),
            "findings_tags": rec.get("findings", []),
            "report_text": text,
            "image_filename": rec.get("image_filename", ""),
        }
        for rec, text in zip(valid_records, texts)
    ]

    with open(output_dir / "report_metadata.json", "w", encoding="utf-8") as f:
        json.dump(report_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Report embeddings saved — shape: %s | Records: %d",
        embeddings.shape,
        len(report_metadata),
    )


if __name__ == "__main__":
    main()
