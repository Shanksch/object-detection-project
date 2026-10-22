"""
Model Manager
==============
Handles YOLOv5 model loading, caching, and warm-up.
Thread-safe singleton with GPU auto-detection.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config.settings import Settings, get_settings
from app.core.exceptions import DetectionError, ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages YOLOv5 model lifecycle: loading, caching, inference, and warm-up.

    This replaces the subprocess-based detection in the original app_clean.py
    with in-process inference for significantly better performance.
    """

    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None
        self._model_path: Optional[Path] = None
        self._device: str = "cpu"
        self._loaded = False
        self._load_time: Optional[float] = None

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> ModelManager:
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_info(self) -> dict[str, Any]:
        """Return model metadata for health checks."""
        return {
            "loaded": self._loaded,
            "device": self._device,
            "model_path": str(self._model_path) if self._model_path else None,
            "load_time_ms": round(self._load_time * 1000, 1) if self._load_time else None,
        }

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the YOLOv5 model into memory.

        Uses torch.hub.load() for in-process inference instead of subprocess calls.
        Automatically selects the best available device (CUDA > MPS > CPU).

        Args:
            force_reload: Force reload even if already loaded.

        Raises:
            ModelNotFoundError: If the model weights file doesn't exist.
            DetectionError: If model loading fails.
        """
        if self._loaded and not force_reload:
            logger.debug("Model already loaded, skipping")
            return

        model_path = self._settings.get_model_path()
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        self._device = self._settings.resolve_device()
        logger.info("Loading model from %s on device '%s'...", model_path, self._device)

        try:
            import torch
            import pathlib
            import platform

            # Fix for loading WindowsPath on Linux Posix system
            if platform.system() != 'Windows':
                pathlib.WindowsPath = pathlib.PosixPath

            start = time.perf_counter()

            # Load custom YOLOv5 model from local repo
            yolo_dir = self._settings.base_dir / self._settings.yolo_dir
            self._model = torch.hub.load(
                str(yolo_dir),
                "custom",
                path=str(model_path),
                source="local",
                device=self._device,
            )

            # Apply default settings
            self._model.conf = self._settings.confidence_threshold
            self._model.iou = self._settings.iou_threshold
            self._model.max_det = self._settings.max_detections

            self._model_path = model_path
            self._load_time = time.perf_counter() - start
            self._loaded = True

            logger.info(
                "✅ Model loaded successfully in %.1fms on '%s'",
                self._load_time * 1000,
                self._device,
            )

        except ImportError:
            raise DetectionError(
                "PyTorch not installed. Run: pip install torch torchvision"
            )
        except Exception as exc:
            raise DetectionError(
                f"Failed to load model: {exc}",
                detail={"model_path": str(model_path), "device": self._device},
            ) from exc

    def warm_up(self, image_size: int = 416) -> None:
        """
        Run a dummy inference to warm up the model (JIT compilation).

        This prevents the first real request from being 5-10x slower.
        """
        if not self._loaded:
            logger.warning("Cannot warm up — model not loaded")
            return

        logger.info("Warming up model with %dx%d dummy input...", image_size, image_size)
        try:
            dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            self._model(dummy)
            logger.info("✅ Model warm-up complete")
        except Exception as exc:
            logger.warning("Model warm-up failed (non-fatal): %s", exc)

    def predict(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        image_size: Optional[int] = None,
    ) -> Any:
        """
        Run object detection on a numpy image array.

        Args:
            image: Input image as numpy array (H, W, C) in RGB.
            confidence: Override confidence threshold for this request.
            image_size: Override image size for this request.

        Returns:
            YOLOv5 Results object with detections.

        Raises:
            DetectionError: If the model is not loaded or inference fails.
        """
        if not self._loaded or self._model is None:
            raise DetectionError("Model is not loaded. Call load_model() first.")

        try:
            # Apply per-request overrides
            if confidence is not None:
                self._model.conf = confidence
            if image_size is not None:
                self._model.imgsz = image_size

            results = self._model(image)

            # Reset to defaults
            self._model.conf = self._settings.confidence_threshold
            self._model.imgsz = self._settings.image_size

            return results

        except Exception as exc:
            raise DetectionError(
                f"Inference failed: {exc}",
                detail={"image_shape": list(image.shape) if hasattr(image, "shape") else "unknown"},
            ) from exc

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        self._loaded = False
        self._model_path = None
        self._load_time = None

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model unloaded from memory")
