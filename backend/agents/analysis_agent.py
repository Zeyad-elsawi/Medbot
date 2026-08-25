"""
backend/agents/analysis_agent.py
Analysis/synthesis node for LangGraph.
Uses Google Gemini 1.5 Flash for clinical report synthesis — no Ollama required.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.config import get_settings
from .state import ClinicalState

logger = logging.getLogger(__name__)


def synthesize_report(state: ClinicalState) -> dict:
    """
    Analysis node for LangGraph.
    Synthesizes the vision findings and RAG context into a final clinical report.
    Uses Gemini 1.5 Flash — no local GPU or Ollama required.
    """
    settings = get_settings()

    # Gather all state
    symptoms = state.get("symptoms_text", "")
    meds = state.get("current_medications", [])
    xray_findings = state.get("xray_findings", "")

    symptom_matches = "\n".join(state.get("symptom_matches", []))
    literature = "\n".join(state.get("literature_refs", []))
    icd10 = "\n".join(state.get("icd10_codes", []))
    drug_int = "\n".join(state.get("drug_interactions", []))

    prompt = f"""
    You are an expert AI clinical physician assistant.
    Synthesize the following patient data into a structured clinical report.
    
    PATIENT INPUTS:
    - Symptoms: {symptoms}
    - Medications: {', '.join(meds) if meds else 'None'}
    
    RADIOLOGY FINDINGS (Vision AI):
    {xray_findings}
    
    CLINICAL CONTEXT (RAG/MCP):
    - Similar Cases: {symptom_matches}
    - Literature: {literature}
    - ICD-10 Code Match: {icd10}
    - Drug Interactions: {drug_int}
    
    INSTRUCTIONS:
    Write a structured clinical markdown report with the following sections:
    1. **Summary of Findings**: Brief summary of symptoms and X-ray findings.
    2. **Differential Diagnosis**: Based on the context, list top 3 possible differentials. (Do not declare a definitive diagnosis).
    3. **Clinical Codes**: Note the relevant ICD-10 code.
    4. **Safety & Interactions**: Note any drug interactions.
    5. **Next Steps**: Recommended clinical follow-up or tests.
    
    Keep the report professional, objective, and strictly based on the provided context.
    Include a disclaimer at the very bottom: "*Disclaimer: AI-generated report for research demonstration. Not medical advice.*"
    """

    try:
        # Initialize Gemini Reasoning Model
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_reasoning_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,  # Slight creativity for synthesis
        )

        logger.info(f"Synthesizing final report with {settings.gemini_reasoning_model}...")
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])

        report = response.content
        logger.info("Analysis complete.")

        return {
            "final_report": report,
            "differential_dx": ["See final report for differentials."],
            "confidence_level": "medium",
            "next_steps": ["See final report for next steps."],
        }

    except Exception as e:
        logger.error(f"Analysis agent error: {str(e)}")
        return {"error": f"Analysis agent failed: {str(e)}"}
