from functools import lru_cache
from sentence_transformers import SentenceTransformer
from core.config import get_settings

@lru_cache(maxsize=1)
def get_text_embedding_model() -> SentenceTransformer:
    """
    Returns a cached instance of the PubMedBERT SentenceTransformer model.
    """
    settings = get_settings()
    # Using cpu by default for free tier
    return SentenceTransformer(settings.pubmedbert_model, device="cpu")
