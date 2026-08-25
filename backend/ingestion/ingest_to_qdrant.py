"""
backend/ingestion/ingest_to_qdrant.py
Uploads all pre-computed embeddings to Qdrant Cloud.
Creates 3 collections: xray_embeddings, clinical_notes, medical_literature.
Run after embed_images.py, embed_reports.py, and embed_literature.py complete.
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Qdrant vector dimensions per collection
XRAY_DIM = 512      # OpenCLIP ViT-B/32
NOTES_DIM = 768     # PubMedBERT / S-PubMedBert-MS-MARCO
LITERATURE_DIM = 768  # same model

BATCH_UPLOAD_SIZE = 100  # Points per Qdrant upsert batch


def get_qdrant_client(settings) -> QdrantClient:
    """
    Create and verify a Qdrant Cloud client connection.

    Args:
        settings: Application settings with qdrant_url and qdrant_api_key.

    Returns:
        Connected QdrantClient instance.

    Raises:
        ConnectionError: If Qdrant is unreachable.
    """
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        # Verify connection by listing collections
        collections = client.get_collections()
        logger.info(
            "Connected to Qdrant at %s | Existing collections: %d",
            settings.qdrant_url,
            len(collections.collections),
        )
        return client
    except Exception as exc:
        logger.error("Failed to connect to Qdrant: %s", exc)
        raise ConnectionError(f"Qdrant connection failed: {exc}") from exc


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int,
) -> None:
    """
    Create a Qdrant collection if it doesn't already exist.

    Uses cosine distance metric — appropriate for L2-normalized embeddings.
    HNSW index with m=16 and ef_construct=100 for good recall/speed balance.

    Args:
        client: Connected QdrantClient.
        collection_name: Name of the collection to create.
        vector_dim: Dimensionality of vectors in this collection.
    """
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            logger.info("Collection '%s' already exists — skipping creation.", collection_name)
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_dim,
                distance=qdrant_models.Distance.COSINE,
            ),
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
        )
        logger.info("Created collection '%s' (dim=%d, metric=cosine)", collection_name, vector_dim)
    except UnexpectedResponse as exc:
        logger.error("Qdrant error creating collection '%s': %s", collection_name, exc)
        raise


def upload_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: list[qdrant_models.PointStruct],
) -> None:
    """
    Upload points to Qdrant in batches.

    Args:
        client: Connected QdrantClient.
        collection_name: Target collection name.
        points: List of PointStruct objects to upload.
    """
    total = len(points)
    logger.info("Uploading %d points to '%s' in batches of %d...", total, collection_name, BATCH_UPLOAD_SIZE)

    for i in tqdm(range(0, total, BATCH_UPLOAD_SIZE), desc=f"Uploading {collection_name}"):
        batch = points[i : i + BATCH_UPLOAD_SIZE]
        try:
            client.upsert(collection_name=collection_name, points=batch)
        except Exception as exc:
            logger.error("Batch upload failed at index %d: %s", i, exc)
            raise

    logger.info("Upload complete for '%s': %d points.", collection_name, total)


def ingest_xray_embeddings(client: QdrantClient, settings) -> int:
    """
    Ingest X-ray image embeddings into the xray_embeddings collection.

    Returns:
        Number of points uploaded.
    """
    data_dir = Path(settings.openi_data_dir)
    embeddings_path = data_dir / "image_embeddings.npy"
    metadata_path = data_dir / "image_metadata.json"

    if not embeddings_path.exists():
        logger.warning("image_embeddings.npy not found — run embed_images.py first. Skipping.")
        return 0

    embeddings = np.load(embeddings_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(embeddings) == len(metadata), (
        f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) count mismatch"
    )

    create_collection_if_not_exists(client, settings.qdrant_xray_collection, XRAY_DIM)

    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["uid"] or str(i))),
            vector=embeddings[i].tolist(),
            payload={
                "uid": meta["uid"],
                "image_filename": meta["image_filename"],
                "diagnosis_label": meta["diagnosis_label"],
                "findings_tags": meta["findings"],
                "source": "openi",
            },
        )
        for i, meta in enumerate(metadata)
    ]

    upload_in_batches(client, settings.qdrant_xray_collection, points)
    return len(points)


def ingest_clinical_notes(client: QdrantClient, settings) -> int:
    """
    Ingest radiology report embeddings into the clinical_notes collection.

    Returns:
        Number of points uploaded.
    """
    data_dir = Path(settings.openi_data_dir)
    embeddings_path = data_dir / "report_embeddings.npy"
    metadata_path = data_dir / "report_metadata.json"

    if not embeddings_path.exists():
        logger.warning("report_embeddings.npy not found — run embed_reports.py first. Skipping.")
        return 0

    embeddings = np.load(embeddings_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    create_collection_if_not_exists(client, settings.qdrant_notes_collection, NOTES_DIM)

    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["uid"] or str(i))),
            vector=embeddings[i].tolist(),
            payload={
                "uid": meta["uid"],
                "diagnosis_label": meta["diagnosis_label"],
                "findings_tags": meta["findings_tags"],
                "report_text": meta["report_text"][:500],  # Truncate for payload size
                "image_filename": meta["image_filename"],
                "source": "openi",
            },
        )
        for i, meta in enumerate(metadata)
    ]

    upload_in_batches(client, settings.qdrant_notes_collection, points)
    return len(points)


def ingest_literature(client: QdrantClient, settings) -> int:
    """
    Ingest PubMed abstract embeddings into the medical_literature collection.

    Returns:
        Number of points uploaded.
    """
    data_dir = Path(settings.literature_data_dir)
    embeddings_path = data_dir / "literature_embeddings.npy"
    metadata_path = data_dir / "literature_metadata.json"

    if not embeddings_path.exists():
        logger.warning("literature_embeddings.npy not found — run embed_literature.py first. Skipping.")
        return 0

    embeddings = np.load(embeddings_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    create_collection_if_not_exists(client, settings.qdrant_literature_collection, LITERATURE_DIM)

    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["pmid"] or str(i))),
            vector=embeddings[i].tolist(),
            payload={
                "pmid": meta["pmid"],
                "title": meta["title"],
                "abstract": meta["abstract"][:500],
                "journal": meta["journal"],
                "year": meta["year"],
                "source": "pubmed",
            },
        )
        for i, meta in enumerate(metadata)
    ]

    upload_in_batches(client, settings.qdrant_literature_collection, points)
    return len(points)


def verify_collections(client: QdrantClient, settings) -> None:
    """
    Print a summary of all 3 collections after ingestion.

    Args:
        client: Connected QdrantClient.
        settings: Application settings with collection names.
    """
    collections = [
        settings.qdrant_xray_collection,
        settings.qdrant_notes_collection,
        settings.qdrant_literature_collection,
    ]
    logger.info("\n=== Qdrant Collection Summary ===")
    for name in collections:
        try:
            info = client.get_collection(name)
            count = info.points_count
            logger.info("  %-35s | %d points", name, count)
        except Exception as exc:
            logger.warning("  %-35s | ERROR: %s", name, exc)
    logger.info("=================================\n")


def main(verify_only: bool = False) -> None:
    """
    Main ingestion orchestrator.

    Args:
        verify_only: If True, only verify existing collections without uploading.
    """
    settings = get_settings()

    try:
        client = get_qdrant_client(settings)
    except ConnectionError as exc:
        logger.error("Cannot proceed — %s", exc)
        sys.exit(1)

    if verify_only:
        logger.info("--verify-only flag set. Checking existing collections...")
        verify_collections(client, settings)
        return

    total_uploaded = 0

    logger.info("=== Starting Qdrant ingestion pipeline ===")

    xray_count = ingest_xray_embeddings(client, settings)
    total_uploaded += xray_count

    notes_count = ingest_clinical_notes(client, settings)
    total_uploaded += notes_count

    lit_count = ingest_literature(client, settings)
    total_uploaded += lit_count

    logger.info("=== Ingestion complete. Total points uploaded: %d ===", total_uploaded)
    verify_collections(client, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest embeddings into Qdrant Cloud")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing collections without uploading",
    )
    args = parser.parse_args()
    main(verify_only=args.verify_only)
