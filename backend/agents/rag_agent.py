import logging
from .state import ClinicalState
from mcp_server.tools.symptom_lookup import lookup_symptoms, find_similar_cases
from mcp_server.tools.literature_search import search_medical_literature
from mcp_server.tools.icd10_finder import find_icd10_code
from mcp_server.tools.drug_interactions import check_drug_interactions

logger = logging.getLogger(__name__)

async def retrieve_clinical_context(state: ClinicalState) -> dict:
    """
    RAG & Tools node for LangGraph.
    Calls our local tools directly to gather medical context.
    """
    symptoms = state.get("symptoms_text", "")
    medications = state.get("current_medications", [])
    
    if not symptoms:
        logger.warning("No symptoms provided. Skipping RAG retrieval.")
        return {}
        
    logger.info("Retrieving medical context via local tools...")
    
    try:
        # 1. Lookup Symptoms & Top Similar Cases
        symptom_list = [s.strip() for s in symptoms.split(",")]
        symptom_matches = lookup_symptoms(symptom_list)
        similar_cases = find_similar_cases(symptoms, top_k=3)
        
        # 2. Search Medical Literature
        literature_refs = search_medical_literature(symptoms, top_k=3)
        
        # 3. Find ICD-10 Code
        icd10_codes = find_icd10_code(symptoms)
        
        # 4. Check Drug Interactions
        drug_interactions = ""
        if medications and len(medications) >= 2:
            drug_interactions = check_drug_interactions(medications)
            
        logger.info("Successfully retrieved medical context.")
        
        return {
            "symptom_matches": [symptom_matches],
            "similar_cases": similar_cases,
            "literature_refs": [literature_refs],
            "icd10_codes": [icd10_codes],
            "drug_interactions": [drug_interactions] if drug_interactions else []
        }
        
    except Exception as e:
        logger.error(f"Error in RAG Agent: {e}")
        return {"error": f"RAG Agent failed: {str(e)}"}
