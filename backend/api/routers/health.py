"""
backend/api/routers/health.py
GET /api/health endpoint.
Returns system status: API running, Qdrant connection, model availability.
"""

import logging
from typing import Any

from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from api.schemas.response import HealthResponse
from core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the operational status of the API and its dependencies.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Perform a health check on all system dependencies.

    Checks:
    - API is running (trivially true if this returns)
    - Qdrant Cloud is reachable and responding
    - Environment is correctly configured

    Returns:
        HealthResponse with status and dependency states.
    """
    settings = get_settings()
    qdrant_ok = False

    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        client.get_collections()
        qdrant_ok = True
        logger.info("Health check: Qdrant connection OK")
    except Exception as exc:
        logger.warning("Health check: Qdrant unreachable — %s", exc)

    overall_status = "ok" if qdrant_ok else "degraded"

    return HealthResponse(
        status=overall_status,
        models_loaded=False,  # Will be True once model loaders are built in Phase 3
        qdrant_connected=qdrant_ok,
        environment=settings.environment,
    )
