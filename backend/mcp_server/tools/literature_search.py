from mcp_server.server import mcp
from retrieval.qdrant_client import get_qdrant_client
from models.embeddings import get_text_embedding_model
from core.config import get_settings

@mcp.tool()
def search_medical_literature(query: str, top_k: int = 5) -> str:
    """
    Search PubMed abstracts stored in the Qdrant vector database.
    Use when you need evidence-based references for findings.
    Returns relevant abstracts with citations and similarity scores.
    """
    settings = get_settings()
    client = get_qdrant_client()
    model = get_text_embedding_model()
    
    # Generate embedding for the query
    vector = model.encode(query).tolist()
    
    try:
        results = client.search(
            collection_name=settings.qdrant_literature_collection,
            query_vector=vector,
            limit=top_k
        )
        
        if not results:
            return f"No medical literature found matching: '{query}'"
            
        formatted_results = []
        for i, hit in enumerate(results, 1):
            title = hit.payload.get("title", "Unknown Title")
            authors = hit.payload.get("authors", "Unknown Authors")
            pub_date = hit.payload.get("pub_date", "Unknown Date")
            abstract = hit.payload.get("abstract", "")
            pmid = hit.payload.get("pmid", "")
            score = round(hit.score, 3)
            
            citation = f"[{i}] {title}\nAuthors: {authors} ({pub_date}) | PMID: {pmid} | Relevance: {score}\nAbstract: {abstract}\n"
            formatted_results.append(citation)
            
        return "Medical Literature Search Results:\n\n" + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching medical literature: {str(e)}"
