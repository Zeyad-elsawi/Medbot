import os
import json
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from core.config import get_settings
from report.pdf_builder import create_pdf

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

@router.get(
    "/report/{report_id}/pdf",
    summary="Download PDF Report",
    description="Generates and downloads a medical-grade PDF of the clinical AI analysis.",
    tags=["Report"],
)
async def get_report_pdf(report_id: str):
    """
    Retrieves the saved report JSON payload and generates a PDF on the fly.
    """
    pdfs_dir = settings.pdf_output_dir
    report_path = os.path.join(pdfs_dir, f"{report_id}.json")
    
    if not os.path.exists(report_path):
        logger.error(f"Report not found: {report_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found."
        )
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        pdf_buffer = create_pdf(payload)
        
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=clinical_report_{report_id[:8]}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF report."
        )
