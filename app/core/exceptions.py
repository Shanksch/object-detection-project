"""
Application Exception Hierarchy
================================
Clean, typed exceptions replacing the legacy SignException.
Each exception carries an HTTP status code for automatic API error responses.
"""

from __future__ import annotations

from typing import Any, Optional


class AppException(Exception):
    """Base application exception with HTTP status code and structured detail."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        status_code: int = 500,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API error responses."""
        payload: dict[str, Any] = {
            "error": True,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


# ── Detection Errors ────────────────────────────────────────────────────

class DetectionError(AppException):
    """Raised when object detection inference fails."""

    def __init__(self, message: str = "Object detection failed", **kwargs: Any) -> None:
        super().__init__(message, status_code=500, **kwargs)


class ModelNotFoundError(AppException):
    """Raised when the model weights file cannot be located."""

    def __init__(self, model_path: str = "") -> None:
        detail = {"model_path": model_path} if model_path else {}
        super().__init__(
            message=f"Model weights not found: {model_path}" if model_path else "Model weights not found",
            status_code=503,
            detail=detail,
        )


# ── Image Errors ────────────────────────────────────────────────────────

class ImageValidationError(AppException):
    """Raised when an uploaded image fails validation (size, format, corruption)."""

    def __init__(self, message: str = "Invalid image", **kwargs: Any) -> None:
        super().__init__(message, status_code=400, **kwargs)


# ── Training Errors ─────────────────────────────────────────────────────

class TrainingError(AppException):
    """Raised when the training pipeline encounters an error."""

    def __init__(self, message: str = "Training pipeline failed", **kwargs: Any) -> None:
        super().__init__(message, status_code=500, **kwargs)


class DataIngestionError(TrainingError):
    """Raised when data download or extraction fails."""

    def __init__(self, message: str = "Data ingestion failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class DataValidationError(TrainingError):
    """Raised when dataset validation fails (missing files, bad format)."""

    def __init__(self, message: str = "Data validation failed", **kwargs: Any) -> None:
        super().__init__(message, status_code=422, **kwargs)
