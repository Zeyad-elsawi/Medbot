"""
backend/agents/state.py
Defines the shared LangGraph state TypedDict that flows through every agent node.
All agents read from and write to this single shared state object.
"""

from typing import TypedDict, Annotated, List, Optional
import operator


def merge_errors(left: Optional[str], right: Optional[str]) -> Optional[str]:
    if not left:
        return right
    if not right:
        return left
    return left + " | " + right


class ClinicalState(TypedDict):
    """
    Shared state for the LangGraph clinical analysis pipeline.

    This TypedDict is the single source of truth that passes between
    Vision Agent → RAG Agent → Analysis Agent in the LangGraph graph.
    All fields are optional at creation time and populated as the
    pipeline progresses.
    """

    # ── Inputs ───────────────────────────────────────────────────────────────
    xray_image_path: str          # Local path to uploaded X-ray image
    symptoms_text: str            # Free-text symptom description (or transcribed voice)
    current_medications: List[str] # List of patient's current medications
    patient_age: Optional[int]    # Optional demographic — not stored
    patient_sex: Optional[str]    # Optional demographic — not stored

    # ── Vision Agent outputs ──────────────────────────────────────────────────
    xray_findings: str            # Structured text findings from LLaVA-Med
    image_embedding: List[float]  # OpenCLIP ViT-B/32 embedding vector (512-dim)

    # ── RAG Agent outputs ─────────────────────────────────────────────────────
    similar_cases: List[dict]     # Top-K historical cases from Qdrant xray_embeddings
    literature_refs: List[dict]   # Relevant PubMed abstracts from Qdrant
    symptom_matches: List[dict]   # Matched conditions from clinical_notes collection

    # ── MCP Tool outputs ──────────────────────────────────────────────────────
    icd10_codes: List[str]        # ICD-10-CM codes from find_icd10_code MCP tool
    drug_interactions: List[str]  # Interaction warnings from check_drug_interactions

    # ── Analysis Agent outputs ────────────────────────────────────────────────
    differential_dx: List[str]    # Differential diagnoses (NOT "diagnosis")
    confidence_level: str         # "high" | "medium" | "low"
    next_steps: List[str]         # Recommended clinical next steps

    # ── Final output ──────────────────────────────────────────────────────────
    final_report: str             # Full structured report text
    report_pdf_path: str          # Absolute path to generated PDF on disk

    # ── Control ───────────────────────────────────────────────────────────────
    # Annotated with operator.add so messages accumulate across parallel nodes
    messages: Annotated[List[str], operator.add]
    error: Annotated[Optional[str], merge_errors] # Set if any node fails — pipeline routes to error handler
