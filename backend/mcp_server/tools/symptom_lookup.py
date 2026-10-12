import json
import logging
from pathlib import Path
from typing import List
import numpy as np

from retrieval.qdrant_client import get_qdrant_client
from models.embeddings import get_text_embedding_model
from core.config import get_settings

logger = logging.getLogger(__name__)

_OPENI_CACHE = None

def _get_local_openi_data():
    global _OPENI_CACHE
    if _OPENI_CACHE is not None:
        return _OPENI_CACHE
    base_dir = Path(__file__).resolve().parents[2] / "data" / "openi"
    emb_path = base_dir / "report_embeddings.npy"
    meta_path = base_dir / "report_metadata.json"
    if emb_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        embeddings = np.load(emb_path)
        _OPENI_CACHE = (embeddings, metadata)
        return _OPENI_CACHE
    return None, None


def find_similar_cases(query: str, top_k: int = 3) -> List[dict]:
    """
    Retrieve top-K structured historical cases using vector search.
    Tries Qdrant Cloud first, then falls back to local Open-I embeddings.
    """
    settings = get_settings()
    model = get_text_embedding_model()
    
    # 1. Try Qdrant Cloud
    try:
        client = get_qdrant_client()
        vector = model.encode(query).tolist()
        results = client.search(
            collection_name=settings.qdrant_notes_collection,
            query_vector=vector,
            limit=top_k
        )
        if results:
            cases = []
            for hit in results:
                cases.append({
                    "case_id": str(hit.id),
                    "diagnosis": hit.payload.get("diagnosis", "Unknown Condition"),
                    "similarity_score": round(float(hit.score), 3),
                    "key_findings": hit.payload.get("report_text", hit.payload.get("diagnosis", ""))[:200],
                    "description": hit.payload.get("diagnosis", "")
                })
            return cases
    except Exception as e:
        logger.warning(f"Qdrant query failed ({e}), using local Open-I vector index fallback.")

    # 2. Local Open-I Vector Fallback
    try:
        embeddings, metadata = _get_local_openi_data()
        if embeddings is not None and metadata is not None:
            qv = model.encode(query)
            q_norm = np.linalg.norm(qv)
            if q_norm > 0:
                qv = qv / q_norm
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(emb_norm, qv)
            top_indices = np.argsort(sims)[::-1][:top_k]

            cases = []
            for idx in top_indices:
                item = metadata[idx]
                score = round(float(sims[idx]), 3)
                score = max(0.0, min(1.0, score))
                diag = item.get("diagnosis_label", "Chest Abnormality")
                findings = item.get("report_text", diag)
                cases.append({
                    "case_id": item.get("uid", f"CXR-{idx+1}"),
                    "diagnosis": diag,
                    "similarity_score": score,
                    "key_findings": findings[:200],
                    "description": diag
                })
            return cases
    except Exception as e:
        logger.error(f"Local vector search error: {e}")

    return []


def lookup_symptoms(symptom_list: List[str]) -> str:
    """
    Given a list of clinical symptoms, return associated conditions
    and their likelihood for differential diagnosis support.
    """
    if not symptom_list:
        return "No symptoms provided to lookup."
        
    query = ", ".join(symptom_list)
    cases = find_similar_cases(query, top_k=5)
    
    if not cases:
        return f"No related clinical conditions found for symptoms: {query}"
        
    formatted = []
    for c in cases:
        score = c.get("similarity_score", 0.8)
        confidence = "High" if score > 0.85 else "Medium" if score > 0.65 else "Low"
        formatted.append(f"- {c['diagnosis']} (Case: {c['case_id']}, Relevance: {score}, Similarity: {confidence})")
        
    return f"Conditions associated with '{query}':\n\n" + "\n".join(formatted)
