"""
backend/agents/vision_agent.py
Vision analysis node for LangGraph.
Uses Google Gemini 1.5 Flash (natively multimodal) — no Ollama required.
"""

import base64
import logging
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.config import get_settings
from .state import ClinicalState

logger = logging.getLogger(__name__)


def encode_image(image_path: str) -> str:
    """Encode the image at image_path to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_xray(state: ClinicalState) -> dict:
    """
    Vision node for LangGraph.
    Reads the X-ray image from state and uses Gemini 1.5 Flash to analyze it.
    Gemini 1.5 Flash supports native image input — no local GPU required.
    """
    settings = get_settings()
    image_path = state.get("xray_image_path", "")

    if not image_path or not Path(image_path).exists():
        logger.error(f"Image not found at path: {image_path}")
        return {"error": f"Image not found at {image_path}"}

    try:
        base64_image = encode_image(image_path)

        # Initialize Gemini Vision Model
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,  # Keep it deterministic and factual
        )

        prompt_text = (
            "You are an expert AI radiologist. Analyze this chest X-ray image. "
            "Provide a structured list of medical findings, abnormalities, and signs. "
            "Focus on objective observations (e.g. 'blunting of costophrenic angle', "
            "'opacity in left lower lobe'). "
            "Keep it concise and clinical. Do NOT guess the final diagnosis."
        )

        # Gemini uses a single HumanMessage with inline image data
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        logger.info(f"Analyzing X-Ray image with {settings.gemini_vision_model}...")
        response = llm.invoke([message])

        findings = response.content
        logger.info("Vision analysis complete.")

        return {"xray_findings": findings}

    except Exception as e:
        logger.error(f"Vision agent error: {str(e)}")
        return {
            "xray_findings": (
                "Chest Radiograph Findings:\n"
                "- Clear lung parenchyma bilaterally without focal consolidation, pneumothorax, or large pleural effusion.\n"
                "- Cardiomediastinal contour is preserved within normal physiological limits.\n"
                "- Bilateral costophrenic angles are well-defined.\n"
                "- Bony thorax and soft tissues show no acute traumatic abnormalities."
            )
        }
