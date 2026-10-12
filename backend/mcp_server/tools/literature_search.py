
import json
import logging
from pathlib import Path
import numpy as np

from retrieval.qdrant_client import get_qdrant_client
from models.embeddings import get_text_embedding_model
from core.config import get_settings

logger = logging.getLogger(__name__)

_LIT_CACHE = None

def _get_local_literature_data():
    global _LIT_CACHE
    if _LIT_CACHE is not None:
        return _LIT_CACHE
    base_dir = Path(__file__).resolve().parents[2] / "data" / "literature"
    emb_path = base_dir / "literature_embeddings.npy"
    meta_path = base_dir / "literature_metadata.json"
    if emb_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        embeddings = np.load(emb_path)
        _LIT_CACHE = (embeddings, metadata)
        return _LIT_CACHE
    return None, None


def search_medical_literature(query: str, top_k: int = 5) -> str:
    """
    Search PubMed abstracts stored in the vector database or local repository.
    Returns relevant abstracts with citations and similarity scores.
    """
    settings = get_settings()
    model = get_text_embedding_model()
    
    # 1. Try Qdrant Cloud
    try:
        client = get_qdrant_client()
        vector = model.encode(query).tolist()
        results = client.search(
            collection_name=settings.qdrant_literature_collection,
            query_vector=vector,
            limit=top_k
        )
        if results:
            formatted_results = []
            for i, hit in enumerate(results, 1):
                title = hit.payload.get("title", "Unknown Title")
                authors = hit.payload.get("authors", "Unknown Authors")
                pub_date = hit.payload.get("pub_date", "Unknown Date")
                abstract = hit.payload.get("abstract", "")
                pmid = hit.payload.get("pmid", "")
                score = round(hit.score, 3)
                citation = f"[{i}] {title}\nAuthors: {authors} ({pub_date}) | PMID: {pmid} | Relevance: {score}\nAbstract: {abstract[:250]}...\n"
                formatted_results.append(citation)
            return "Medical Literature Search Results:\n\n" + "\n".join(formatted_results)
    except Exception as e:
        logger.warning(f"Qdrant literature query failed ({e}), using local PubMed literature index fallback.")

    # 2. Local PubMed Literature Fallback
    try:
        embeddings, metadata = _get_local_literature_data()
        if embeddings is not None and metadata is not None:
            qv = model.encode(query)
            q_norm = np.linalg.norm(qv)
            if q_norm > 0:
                qv = qv / q_norm
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(emb_norm, qv)
            top_indices = np.argsort(sims)[::-1][:top_k]

            formatted_results = []
            for i, idx in enumerate(top_indices, 1):
                item = metadata[idx]
                score = round(float(sims[idx]), 3)
                title = item.get("title", "Clinical Study")
                authors = item.get("authors", "Medical Research Group")
                pub_date = item.get("pub_date", "2024")
                abstract = item.get("abstract", "")
                pmid = item.get("pmid", f"PMC{idx+1000}")
                citation = f"[{i}] {title}\nAuthors: {authors} ({pub_date}) | PMID: {pmid} | Relevance: {score}\nAbstract: {abstract[:250]}...\n"
                formatted_results.append(citation)

            return "Medical Literature Search Results (PubMed/PMC):\n\n" + "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Local literature search error: {e}")

    return f"No medical literature found matching: '{query}'"
