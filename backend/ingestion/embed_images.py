"""
backend/ingestion/embed_images.py
Generates OpenCLIP ViT-B/32 embeddings for all chest X-ray images.
Saves embeddings as a numpy file alongside metadata for Qdrant ingestion.
Run after download_openi.py completes.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # OpenCLIP ViT-B/32 output dimension


def load_clip_model(settings) -> tuple:
    """
    Load OpenCLIP ViT-B/32 model and preprocessing transforms.

    Uses CPU for local ingestion — sufficient for embedding generation.
    Model is downloaded on first run and cached by open_clip.

    Args:
        settings: Application settings with model name/pretrained config.

    Returns:
        Tuple of (model, preprocess_fn, device).

    Raises:
        RuntimeError: If model fails to load.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading OpenCLIP %s (%s) on %s...", settings.openclip_model, settings.openclip_pretrained, device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.openclip_model,
            pretrained=settings.openclip_pretrained,
        )
        model = model.to(device)
        model.eval()

        logger.info("OpenCLIP model loaded successfully.")
        return model, preprocess, device
    except Exception as exc:
        logger.error("Failed to load OpenCLIP model: %s", exc)
        raise RuntimeError(f"OpenCLIP load failed: {exc}") from exc


def embed_image_batch(
    model,
    preprocess,
    device: str,
    image_paths: list[Path],
) -> np.ndarray:
    """
    Generate embeddings for a batch of images.

    Args:
        model: Loaded OpenCLIP model.
        preprocess: Image preprocessing transform from open_clip.
        device: 'cuda' or 'cpu'.
        image_paths: List of image file Paths to embed.

    Returns:
        numpy array of shape (N, 512) — L2-normalized embeddings.
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", path.name, exc)
            images.append(torch.zeros(3, 224, 224))  # Zero vector placeholder

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

    return features.cpu().numpy().astype(np.float32)


def main() -> None:
    """
    Main embedding pipeline for all OpenI X-ray images.

    Reads images from data/openi/images/,
    processes in batches of settings.openi_batch_size,
    saves embeddings + record IDs to data/openi/image_embeddings.npy
    and data/openi/image_ids.json.
    """
    settings = get_settings()

    images_dir = Path(settings.openi_data_dir) / "images"
    reports_dir = Path(settings.openi_data_dir) / "reports"
    metadata_path = reports_dir / "metadata.json"

    if not images_dir.exists():
        logger.error("Images directory not found: %s — run download_openi.py first", images_dir)
        sys.exit(1)

    if not metadata_path.exists():
        logger.error("Metadata not found: %s — run download_openi.py first", metadata_path)
        sys.exit(1)

    # Load metadata to map filenames → record UIDs
    with open(metadata_path, encoding="utf-8") as f:
        records = json.load(f)

    filename_to_record = {r["image_filename"]: r for r in records if r.get("image_filename")}

    # Collect all valid image paths
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    image_files = [p for p in image_files if p.name in filename_to_record]

    if not image_files:
        logger.error("No valid images found in %s matching metadata.", images_dir)
        sys.exit(1)

    logger.info("Found %d images to embed.", len(image_files))

    # Load model
    model, preprocess, device = load_clip_model(settings)

    # Process in batches
    batch_size = settings.openi_batch_size
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Embedding X-rays", unit="batch"):
        batch_paths = image_files[i : i + batch_size]
        batch_embeddings = embed_image_batch(model, preprocess, device, batch_paths)

        all_embeddings.append(batch_embeddings)

        for path in batch_paths:
            rec = filename_to_record[path.name]
            all_ids.append(rec["uid"] or path.stem)
            all_metadata.append(
                {
                    "uid": rec["uid"],
                    "image_filename": path.name,
                    "diagnosis_label": rec.get("diagnosis_label", "unknown"),
                    "findings": rec.get("findings", []),
                }
            )

    # Concatenate and save
    embeddings_array = np.vstack(all_embeddings)
    output_dir = Path(settings.openi_data_dir)

    np.save(output_dir / "image_embeddings.npy", embeddings_array)

    with open(output_dir / "image_ids.json", "w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    with open(output_dir / "image_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(
        "Embeddings saved — shape: %s | IDs: %d | Files: %s",
        embeddings_array.shape,
        len(all_ids),
        output_dir,
    )


if __name__ == "__main__":
    main()
