"""
backend/api/main.py
FastAPI application entry point.
Configures CORS, mounts all routers, sets up structured logging.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import health, analyze, report
from api.schemas.response import ErrorResponse
from core.config import get_settings

# ── Logging Setup ──────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Runs startup logic before yielding, shutdown logic after.
    Model loading will be added here in Phase 3.
    """
    logger.info("Starting Multimodal Clinical AI Copilot — environment: %s", settings.environment)
    logger.info("Qdrant URL: %s", settings.qdrant_url)
    logger.info("CORS origins: %s", settings.cors_origins_list)
    yield
    logger.info("Shutting down Clinical AI Copilot")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Clinical AI Copilot",
    description=(
        "AI-assisted chest X-ray analysis system. "
        "NOT a medical device. NOT for clinical use. "
        "Research and portfolio demonstration only."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Global Exception Handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler.
    Ensures raw Python tracebacks are never exposed to the frontend.
    Logs the full exception internally for debugging.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="An internal error occurred. Please try again.",
            code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(report.router, prefix="/api")


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect root to Swagger docs."""
    return JSONResponse({"message": "Clinical AI Copilot API. See /docs for Swagger UI."})
