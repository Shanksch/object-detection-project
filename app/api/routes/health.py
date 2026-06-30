"""
Health & UI Routes
===================
Health check endpoint and static UI serving.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.api.schemas import HealthResponse
from app.config.settings import get_settings
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# App start time for uptime calculation
_start_time = time.time()


@router.get(
    "/",
    response_class=HTMLResponse,
    summary="Main UI",
    include_in_schema=False,
)
async def serve_ui() -> FileResponse:
    """Serve the main object detection UI."""
    settings = get_settings()

    # Try static/index.html first, then fall back to templates/modern_ui.html
    candidates = [
        settings.base_dir / "static" / "index.html",
        settings.base_dir / "templates" / "modern_ui.html",
    ]

    for path in candidates:
        if path.exists():
            return FileResponse(str(path), media_type="text/html")

    return HTMLResponse(
        content="<h1>Object Detection Studio</h1><p>UI not found. Place index.html in /static/</p>",
        status_code=200,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server health status, model info, and uptime.",
)
async def health_check() -> dict:
    """Health check endpoint with model information."""
    model = ModelManager.get_instance()
    settings = get_settings()

    return {
        "status": "healthy" if model.is_loaded else "degraded",
        "message": "Object Detection API is running!",
        "model": {
            **model.model_info,
            "weights": settings.model_weights,
            "confidence_threshold": settings.confidence_threshold,
            "device": model.device,
        },
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
