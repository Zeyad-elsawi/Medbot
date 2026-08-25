from typing import List
from mcp_server.server import mcp
from retrieval.qdrant_client import get_qdrant_client
from models.embeddings import get_text_embedding_model
from core.config import get_settings

@mcp.tool()
def lookup_symptoms(symptom_list: List[str]) -> str:
    """
    Given a list of clinical symptoms, return associated conditions
    and their likelihood for differential diagnosis support.
    Use before generating the differential diagnoses section.
    """
    if not symptom_list:
        return "No symptoms provided to lookup."
        
    query = ", ".join(symptom_list)
    settings = get_settings()
    client = get_qdrant_client()
    model = get_text_embedding_model()
    
    # Generate embedding for the combined symptoms
    vector = model.encode(query).tolist()
    
    try:
        # Search against clinical notes collection
        results = client.search(
            collection_name=settings.qdrant_notes_collection,
            query_vector=vector,
            limit=5
        )
        
        if not results:
            return f"No related clinical conditions found for symptoms: {query}"
            
        # Aggregate diagnoses from the similar notes
        diagnoses = {}
        for hit in results:
            diag_label = hit.payload.get("diagnosis", "Unknown Condition")
            if diag_label not in diagnoses:
                diagnoses[diag_label] = []
            diagnoses[diag_label].append(hit.score)
            
        # Format the output
        formatted = []
        for diag, scores in sorted(diagnoses.items(), key=lambda x: max(x[1]), reverse=True):
            avg_score = sum(scores) / len(scores)
            confidence = "High" if avg_score > 0.8 else "Medium" if avg_score > 0.6 else "Low"
            formatted.append(f"- {diag.title()} (Matches: {len(scores)}, Avg Relevance: {round(avg_score, 3)}, Similarity: {confidence})")
            
        return f"Conditions associated with '{query}':\n\n" + "\n".join(formatted)
        
    except Exception as e:
        return f"Error looking up symptoms: {str(e)}"
