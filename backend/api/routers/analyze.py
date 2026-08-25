import json
import logging
import uuid
import tempfile
import os
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.schemas.request import AnalysisRequest
from api.schemas.response import AnalysisResponse
from core.config import get_settings
from agents.orchestrator import clinical_agent
from agents.state import ClinicalState

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

def format_sse(type_: str, message: str = None, step: int = None, total_steps: int = 7, payload: dict = None) -> str:
    """Format a Server-Sent Event message."""
    data = {"type": type_, "total_steps": total_steps}
    if message:
        data["message"] = message
    if step is not None:
        data["step"] = step
    if payload is not None:
        data["payload"] = payload
    return f"data: {json.dumps(data)}\n\n"

@router.post(
    "/analyze",
    summary="Analyze Chest X-ray (Streaming)",
    description="Accepts a chest X-ray image and patient symptoms. Streams progress via SSE.",
    tags=["Analysis"],
)
async def analyze(
    image: UploadFile = File(..., description="Chest X-ray image (JPG, PNG, DICOM)"),
    symptoms: str = Form(..., min_length=10, max_length=2000),
    current_medications: str = Form(default=""),
    patient_age: int | None = Form(default=None),
    patient_sex: str | None = Form(default=None),
):
    """
    Main analysis endpoint using SSE to stream LangGraph execution progress.
    """
    # Validate image format
    allowed_types = {"image/jpeg", "image/png", "image/dicom", "application/octet-stream"}
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Accepted: JPG, PNG, DICOM",
        )

    # Save to temp file so Vision Agent can access it
    content = await image.read()
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{image.filename}")
    with open(temp_path, "wb") as f:
        f.write(content)

    meds_list = [m.strip() for m in current_medications.split(",")] if current_medications else []

    initial_state: ClinicalState = {
        "xray_image_path": temp_path,
        "symptoms_text": symptoms,
        "current_medications": meds_list,
        "patient_age": patient_age,
        "patient_sex": patient_sex
    }

    async def event_generator():
        try:
            yield format_sse("status", "Vision agent analyzing X-ray...", 1)
            
            final_state = None
            
            # Run LangGraph streaming
            async for chunk in clinical_agent.astream(initial_state):
                for node_name, node_state in chunk.items():
                    if node_name == "vision":
                        yield format_sse("status", "Extracting imaging findings...", 2)
                    elif node_name == "rag":
                        yield format_sse("status", "Searching similar cases...", 3)
                        yield format_sse("status", "Retrieving medical literature...", 4)
                        yield format_sse("status", "Checking drug interactions...", 5)
                    elif node_name == "analysis":
                        yield format_sse("status", "Generating report...", 6)
                        final_state = node_state

            if final_state is None:
                # Fallback if astream didn't yield final state for some reason
                final_state = await clinical_agent.ainvoke(initial_state)

            yield format_sse("status", "Finalizing structured report...", 7)

            # Map to final payload
            report_id = str(uuid.uuid4())
            payload = {
                "report_id": report_id,
                "imaging_findings": final_state.get("xray_findings", "No findings."),
                "similar_cases": [{"case_id": "N/A", "description": sc, "similarity_score": 0.0} for sc in final_state.get("symptom_matches", [])],
                "differential_dx": final_state.get("differential_dx", []),
                "icd10_codes": final_state.get("icd10_codes", []),
                "next_steps": final_state.get("next_steps", []),
                "confidence_level": final_state.get("confidence_level", "low"),
                "disclaimer": "*Disclaimer: AI-generated report for research demonstration. Not medical advice.*",
                "pdf_download_url": f"/api/report/{report_id}/pdf",
                "full_markdown_report": final_state.get("final_report", "")
            }
            
            # Save payload to disk for PDF retrieval later
            pdfs_dir = os.path.join(settings.pdf_output_dir)
            os.makedirs(pdfs_dir, exist_ok=True)
            report_path = os.path.join(pdfs_dir, f"{report_id}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            
            yield format_sse("result", payload=payload)
            yield format_sse("done")

        except Exception as e:
            logger.error(f"Error during streaming analysis: {e}")
            yield format_sse("error", message=str(e))
        finally:
            # Cleanup temp image
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
