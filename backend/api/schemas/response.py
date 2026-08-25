"""
backend/api/schemas/response.py
Pydantic v2 response models for all FastAPI endpoints.
These schemas are the source of truth — Angular TypeScript interfaces must match exactly.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

MANDATORY_DISCLAIMER = (
    "⚠️ IMPORTANT DISCLAIMER: This system is a research and portfolio demonstration tool only. "
    "It is NOT a medical device and NOT approved for clinical use. "
    "All outputs are AI-generated and must NOT be used for actual medical diagnosis, "
    "treatment decisions, or patient care. "
    "Always consult a qualified healthcare professional for medical advice."
)


class SimilarCase(BaseModel):
    """A single similar historical case retrieved from Qdrant."""

    case_id: str = Field(description="Unique identifier of the historical case")
    diagnosis: str = Field(description="Diagnosis label from dataset metadata")
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity score (0-1)"
    )
    key_findings: str = Field(description="Key imaging findings from the historical report")


class AnalysisResponse(BaseModel):
    """
    Response model for POST /api/analyze.
    Returned to Angular frontend after full multi-agent pipeline completes.
    IMPORTANT: This schema must stay in sync with clinical.models.ts in Angular.
    """

    report_id: str = Field(description="UUID for this analysis session")
    imaging_findings: str = Field(
        description="Structured imaging findings from Vision Agent (LLaVA-Med)"
    )
    similar_cases: List[SimilarCase] = Field(
        default_factory=list,
        description="Top-3 similar historical cases from Qdrant",
    )
    differential_dx: List[str] = Field(
        default_factory=list,
        description="Possible differential diagnoses (NOT definitive diagnosis)",
    )
    icd10_codes: List[str] = Field(
        default_factory=list,
        description="ICD-10-CM codes corresponding to differential findings",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested clinical next steps for physician consideration",
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="AI confidence in findings based on image quality and symptom clarity"
    )
    disclaimer: str = Field(
        default=MANDATORY_DISCLAIMER,
        description="Mandatory medical disclaimer — always present",
    )
    pdf_download_url: str = Field(
        description="Relative URL to download the PDF report: /api/report/{report_id}/pdf"
    )


class HealthResponse(BaseModel):
    """Response model for GET /api/health."""

    status: Literal["ok", "degraded", "error"]
    models_loaded: bool
    qdrant_connected: bool
    environment: str


class ErrorResponse(BaseModel):
    """Structured error response — no raw Python tracebacks exposed."""

    error: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")
