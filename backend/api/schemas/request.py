"""
backend/api/schemas/request.py
Pydantic v2 request models for all FastAPI endpoints.
Validates all incoming data before it reaches any business logic.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class AnalysisRequest(BaseModel):
    """
    Request model for POST /api/analyze.
    Image is received as multipart/form-data separately — these are the
    text fields that accompany the image upload.
    """

    symptoms: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Free-text description of patient symptoms",
        examples=["Patient reports persistent cough, fever for 5 days, shortness of breath"],
    )
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age in years (optional, not stored)",
    )
    patient_sex: Optional[str] = Field(
        default=None,
        description="Patient biological sex: M or F (optional, not stored)",
    )

    @field_validator("patient_sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        """Normalize and validate patient sex field."""
        if v is None:
            return v
        normalized = v.upper().strip()
        if normalized not in ("M", "F", "MALE", "FEMALE"):
            raise ValueError("patient_sex must be 'M' or 'F'")
        return normalized[0]  # Store as single char


class VoiceTranscriptionRequest(BaseModel):
    """Request model for optional voice transcription endpoint."""

    audio_format: str = Field(
        default="webm",
        description="Audio file format: webm, mp3, wav",
    )
