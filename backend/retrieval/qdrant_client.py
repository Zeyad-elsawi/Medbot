from functools import lru_cache
from qdrant_client import QdrantClient
from core.config import get_settings

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Returns a cached instance of the Qdrant client.
    """
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
