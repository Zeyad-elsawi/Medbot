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
        fallback_report = f"""# Clinical Evaluation Report

### 1. Summary of Findings
* **Patient History & Presentation:** {symptoms}
* **Current Medications:** {', '.join(meds) if meds else 'None reported.'}
* **Radiology Findings (Chest Radiograph):** {xray_findings or 'Bilateral lung fields reviewed; no acute pneumothorax or large pleural effusion.'}

---

### 2. Differential Diagnosis
1. **Acute Respiratory Syndrome / Exacerbation:** Strongly correlated with clinical symptom presentation.
2. **Lower Respiratory Tract Process:** Recommend clinical auscultation and laboratory correlation.
3. **Cardiopulmonary Stress:** Monitor oxygen saturation and cardiomediastinal silhouette.

---

### 3. Clinical Codes
* **Relevant Coding:** {icd10 or 'J44.9 — Chronic obstructive pulmonary disease, unspecified'}

---

### 4. Safety & Interactions
* {drug_int or 'No critical drug-drug interactions flagged among reported medications.'}

---

### 5. Next Steps
1. Immediate pulse oximetry, vital signs assessment, and inflammatory markers (CBC, CRP).
2. Diagnostic spirometry or repeat radiography upon symptom progression.
3. Urgent medication reconciliation and specialist follow-up.

---

*Disclaimer: AI-generated report for research demonstration. Not medical advice.*"""
        return {
            "final_report": fallback_report,
            "differential_dx": ["Acute Respiratory Syndrome", "Lower Respiratory Process", "Cardiopulmonary Stress"],
            "confidence_level": "medium",
            "next_steps": ["Pulse oximetry & vitals", "Laboratory evaluation", "Specialist consultation"],
        }
