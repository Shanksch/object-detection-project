"""
Detection Routes
=================
Endpoints for object detection: JSON body and file upload.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.api.schemas import DetectRequest, DetectResponse, LiveStatusResponse
from app.services.detection_service import DetectionService
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Detection"])

# Service instances (initialized once, reused across requests)
_detection_service: Optional[DetectionService] = None


def _get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


@router.post(
    "/predict",
    response_model=DetectResponse,
    summary="Run object detection (JSON body)",
    description="Submit a base64-encoded image for object detection. Returns annotated image and structured detection data.",
)
async def predict_json(request: DetectRequest) -> dict:
    """
    Object detection via JSON body with base64 image.

    - **image**: Base64-encoded image (required)
    - **confidence**: Override confidence threshold (0.0-1.0)
    - **image_size**: Override detection image size
    """
    service = _get_detection_service()
    result = service.detect_from_base64(
        image_b64=request.image,
        confidence=request.confidence,
        image_size=request.image_size,
    )
    return result.to_dict()


@router.post(
    "/predict/file",
    response_model=DetectResponse,
    summary="Run object detection (file upload)",
    description="Upload an image file for object detection. Supports JPG, PNG, WEBP, BMP.",
)
async def predict_file(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: Optional[float] = Form(default=None, ge=0.0, le=1.0),
    image_size: Optional[int] = Form(default=None, ge=32, le=1280),
) -> dict:
    """
    Object detection via multipart file upload.
    More efficient than base64 — no 33% encoding overhead.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload.jpg"

    service = _get_detection_service()
    result = service.detect_from_upload(
        file_bytes=file_bytes,
        filename=filename,
        confidence=confidence,
        image_size=image_size,
    )
    return result.to_dict()


@router.get(
    "/live",
    response_model=LiveStatusResponse,
    summary="Live camera detection status",
)
async def live_status() -> dict:
    """Returns the status of live camera detection capability."""
    model = ModelManager.get_instance()
    return {
        "status": "ready" if model.is_loaded else "model_not_loaded",
        "message": "Live camera detection is ready. Camera access is handled by the browser.",
        "instructions": "Use the 'Capture & Analyze' button to take snapshots for detection.",
    }
