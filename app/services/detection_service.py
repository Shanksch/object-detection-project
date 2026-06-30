"""
Detection Service
==================
Business logic for object detection: image → inference → structured results.
No filesystem scanning — all results stay in memory.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from app.config.settings import get_settings
from app.core.exceptions import DetectionError
from app.services.image_service import ImageService
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


class DetectionResult:
    """Structured detection result with metadata."""

    def __init__(
        self,
        annotated_image: Image.Image,
        detections: list[dict[str, Any]],
        inference_time_ms: float,
        model_name: str,
        image_size: tuple[int, int],
    ) -> None:
        self.annotated_image = annotated_image
        self.detections = detections
        self.inference_time_ms = inference_time_ms
        self.model_name = model_name
        self.image_size = image_size
        self.count = len(detections)

    def to_dict(self, include_image: bool = True) -> dict[str, Any]:
        """Serialize to API response format."""
        result: dict[str, Any] = {
            "success": True,
            "detections": self.detections,
            "metadata": {
                "count": self.count,
                "inference_time_ms": round(self.inference_time_ms, 1),
                "model": self.model_name,
                "image_size": list(self.image_size),
            },
        }
        if include_image:
            result["image"] = ImageService.encode_to_base64(self.annotated_image)
        return result


class DetectionService:
    """Orchestrates the detection pipeline: decode → infer → format results."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        image_service: Optional[ImageService] = None,
    ) -> None:
        self._model = model_manager or ModelManager.get_instance()
        self._image = image_service or ImageService()
        self._settings = get_settings()

    def detect_from_base64(
        self,
        image_b64: str,
        confidence: Optional[float] = None,
        image_size: Optional[int] = None,
    ) -> DetectionResult:
        """
        Run detection on a base64-encoded image.

        Args:
            image_b64: Base64-encoded image string.
            confidence: Override confidence threshold.
            image_size: Override image size.

        Returns:
            DetectionResult with annotated image and structured detections.
        """
        pil_image = self._image.decode_base64(
            image_b64, max_size_bytes=self._settings.max_file_size
        )
        return self._run_detection(pil_image, confidence, image_size)

    def detect_from_upload(
        self,
        file_bytes: bytes,
        filename: str,
        confidence: Optional[float] = None,
        image_size: Optional[int] = None,
    ) -> DetectionResult:
        """
        Run detection on an uploaded file.

        Args:
            file_bytes: Raw file bytes from multipart upload.
            filename: Original filename.
            confidence: Override confidence threshold.
            image_size: Override image size.

        Returns:
            DetectionResult with annotated image and structured detections.
        """
        pil_image = self._image.decode_upload_bytes(
            file_bytes, filename, max_size_bytes=self._settings.max_file_size
        )
        return self._run_detection(pil_image, confidence, image_size)

    def _run_detection(
        self,
        pil_image: Image.Image,
        confidence: Optional[float],
        image_size: Optional[int],
    ) -> DetectionResult:
        """Core detection logic shared by all input methods."""
        np_image = self._image.pil_to_numpy(pil_image)

        logger.info(
            "Running detection on %dx%d image (conf=%.2f, size=%s)",
            pil_image.width,
            pil_image.height,
            confidence or self._settings.confidence_threshold,
            image_size or self._settings.image_size,
        )

        start = time.perf_counter()
        results = self._model.predict(np_image, confidence=confidence, image_size=image_size)
        inference_time = (time.perf_counter() - start) * 1000  # ms

        # Extract structured detections
        detections = self._parse_detections(results)

        # Get annotated image from YOLOv5 results
        annotated = self._get_annotated_image(results)

        logger.info(
            "✅ Detection complete: %d objects in %.1fms",
            len(detections),
            inference_time,
        )

        return DetectionResult(
            annotated_image=annotated,
            detections=detections,
            inference_time_ms=inference_time,
            model_name=self._settings.model_weights,
            image_size=(pil_image.width, pil_image.height),
        )

    def _parse_detections(self, results: Any) -> list[dict[str, Any]]:
        """Extract structured detection data from YOLOv5 results."""
        detections = []
        try:
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                detections.append({
                    "class": row["name"],
                    "confidence": round(float(row["confidence"]), 4),
                    "bbox": {
                        "x_min": round(float(row["xmin"]), 1),
                        "y_min": round(float(row["ymin"]), 1),
                        "x_max": round(float(row["xmax"]), 1),
                        "y_max": round(float(row["ymax"]), 1),
                    },
                    "class_id": int(row["class"]),
                })
        except Exception as exc:
            logger.warning("Failed to parse detections: %s", exc)
        return detections

    def _get_annotated_image(self, results: Any) -> Image.Image:
        """Convert YOLOv5 annotated results to PIL Image."""
        try:
            rendered = results.render()
            if rendered and len(rendered) > 0:
                return ImageService.numpy_to_pil(rendered[0])
        except Exception as exc:
            logger.warning("Failed to render annotated image: %s", exc)
        # Fallback: return a blank image
        return Image.new("RGB", (416, 416), color=(0, 0, 0))
