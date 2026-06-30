"""
Application Settings — Single Source of Truth
==============================================
Pydantic-based configuration with .env loading, validation, and type safety.
Replaces both config.py and objectDetection/constant/training_pipeline/__init__.py.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    # ── Server ──────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8080, ge=1024, le=65535, description="Server bind port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=1, ge=1, description="Number of Uvicorn workers")

    # ── Paths ───────────────────────────────────────────────────────────
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="Project root directory",
    )
    data_dir: Path = Field(default=Path("data"), description="Runtime data directory")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Training artifacts")
    static_dir: Path = Field(default=Path("static"), description="Static files directory")
    log_dir: Path = Field(default=Path("logs"), description="Log files directory")

    # ── Model ───────────────────────────────────────────────────────────
    yolo_dir: str = Field(default="yolov5", description="YOLOv5 repo directory")
    model_weights: str = Field(default="best.pt", description="Model weights filename")
    image_size: int = Field(default=416, ge=32, le=1280, description="Detection image size")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="NMS IoU threshold"
    )
    max_detections: int = Field(default=1000, ge=1, description="Max detections per image")
    device: str = Field(
        default="auto",
        description="Inference device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )

    # ── Upload / Security ───────────────────────────────────────────────
    max_file_size: int = Field(
        default=10_485_760, ge=1, description="Max upload file size in bytes (10 MB)"
    )
    allowed_extensions: FrozenSet[str] = Field(
        default=frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"}),
        description="Allowed image upload extensions",
    )

    # ── Training Pipeline ───────────────────────────────────────────────
    data_download_url: str = Field(
        default="https://github.com/Shanksch/data/raw/refs/heads/main/object_detection_dataset.zip",
        description="Dataset download URL",
    )
    train_epochs: int = Field(default=25, ge=1, description="Training epochs")
    train_batch_size: int = Field(default=16, ge=1, description="Training batch size")
    pretrained_weights: str = Field(default="yolov5s.pt", description="Pretrained weight file")

    # ── Rate Limiting ───────────────────────────────────────────────────
    rate_limit: str = Field(
        default="30/minute", description="API rate limit (slowapi format)"
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=False, description="Use JSON log format (for production)")
    log_max_bytes: int = Field(default=10_485_760, description="Max log file size")
    log_backup_count: int = Field(default=5, description="Number of rotated log backups")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v_upper

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v == "auto":
            return v
        valid_prefixes = ("cpu", "cuda", "mps")
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(f"device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps', got '{v}'")
        return v

    # ── Computed helpers ────────────────────────────────────────────────
    def resolve_device(self) -> str:
        """Resolve 'auto' to best available device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_model_path(self) -> Path:
        """Full path to the model weights file."""
        return self.base_dir / self.yolo_dir / self.model_weights

    def ensure_directories(self) -> None:
        """Create runtime directories if they don't exist."""
        for d in (self.data_dir, self.artifacts_dir, self.log_dir, self.static_dir):
            resolved = d if d.is_absolute() else self.base_dir / d
            resolved.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
