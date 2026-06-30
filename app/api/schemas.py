"""
API Request/Response Schemas
==============================
Pydantic models for all API endpoints.
Provides automatic validation, serialization, and OpenAPI documentation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Detection Schemas ───────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """Request body for POST /api/v1/predict (JSON mode)."""

    image: str = Field(..., description="Base64-encoded image string")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override confidence threshold"
    )
    image_size: Optional[int] = Field(
        default=None, ge=32, le=1280, description="Override detection image size"
    )


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    """A single detected object."""

    class_name: str = Field(alias="class")
    confidence: float
    bbox: BoundingBox
    class_id: int


class DetectionMetadata(BaseModel):
    """Metadata about the detection run."""

    count: int
    inference_time_ms: float
    model: str
    image_size: list[int]


class DetectResponse(BaseModel):
    """Response body for detection endpoints."""

    success: bool = True
    image: Optional[str] = Field(default=None, description="Base64-encoded annotated image")
    detections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: DetectionMetadata


# ── Training Schemas ────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    """Request body for POST /api/v1/train."""

    epochs: Optional[int] = Field(default=None, ge=1, le=1000, description="Training epochs")
    batch_size: Optional[int] = Field(default=None, ge=1, le=128, description="Training batch size")
    image_size: Optional[int] = Field(default=None, ge=32, le=1280, description="Training image size")


class TrainResponse(BaseModel):
    """Response body for POST /api/v1/train."""

    task_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    """Response body for GET /api/v1/train/{task_id}."""

    task_id: str
    status: str
    progress: float
    message: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    config: dict[str, Any] = Field(default_factory=dict)


# ── Health Schemas ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    message: str
    model: dict[str, Any]
    uptime_seconds: float


# ── Error Schemas ───────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: bool = True
    message: str
    status_code: int
    detail: Optional[dict[str, Any]] = None


# ── Live Camera Schemas ─────────────────────────────────────────────────

class LiveStatusResponse(BaseModel):
    """Response body for GET /api/v1/live."""

    status: str
    message: str
    instructions: str
