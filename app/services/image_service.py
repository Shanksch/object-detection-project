"""
Image Service
==============
Handles image validation, encoding, and decoding.
Replaces the raw utility functions from objectDetection/utils/main_utils.py.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.core.exceptions import ImageValidationError

logger = logging.getLogger(__name__)


class ImageService:
    """Stateless service for image processing operations."""

    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

    @staticmethod
    def decode_base64(
        image_b64: str,
        max_size_bytes: int = 10_485_760,
    ) -> Image.Image:
        """
        Decode a base64 string into a PIL Image.

        Args:
            image_b64: Base64-encoded image string.
            max_size_bytes: Maximum allowed image size in bytes.

        Returns:
            PIL Image object.

        Raises:
            ImageValidationError: If the image is invalid, too large, or corrupted.
        """
        try:
            raw_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            raise ImageValidationError(
                "Invalid base64 encoding", detail={"reason": str(exc)}
            ) from exc

        if len(raw_bytes) > max_size_bytes:
            raise ImageValidationError(
                f"Image exceeds maximum size of {max_size_bytes // (1024 * 1024)} MB",
                detail={"size_bytes": len(raw_bytes), "max_bytes": max_size_bytes},
            )

        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.verify()
            # Re-open after verify (verify leaves the file pointer in a bad state)
            img = Image.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ImageValidationError(
                "Corrupted or unsupported image format",
                detail={"reason": str(exc)},
            ) from exc

        if img.format and img.format.upper() not in ImageService.SUPPORTED_FORMATS:
            raise ImageValidationError(
                f"Unsupported format: {img.format}. Supported: {ImageService.SUPPORTED_FORMATS}"
            )

        logger.debug("Decoded image: %dx%d %s", img.width, img.height, img.format)
        return img

    @staticmethod
    def decode_upload_bytes(
        file_bytes: bytes,
        filename: str,
        max_size_bytes: int = 10_485_760,
    ) -> Image.Image:
        """
        Decode uploaded file bytes into a PIL Image.

        Args:
            file_bytes: Raw file bytes from multipart upload.
            filename: Original filename (for extension validation).
            max_size_bytes: Maximum allowed file size.

        Returns:
            PIL Image object.
        """
        if len(file_bytes) > max_size_bytes:
            raise ImageValidationError(
                f"File exceeds maximum size of {max_size_bytes // (1024 * 1024)} MB",
                detail={"size_bytes": len(file_bytes), "max_bytes": max_size_bytes},
            )

        ext = Path(filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if ext not in allowed:
            raise ImageValidationError(
                f"Unsupported file extension: {ext}. Allowed: {allowed}"
            )

        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            img = Image.open(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ImageValidationError(
                "Corrupted or unreadable image file",
                detail={"filename": filename, "reason": str(exc)},
            ) from exc

        logger.debug("Decoded upload: %s (%dx%d)", filename, img.width, img.height)
        return img

    @staticmethod
    def encode_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
        """
        Encode a PIL Image to a base64 string.

        Args:
            img: PIL Image to encode.
            fmt: Output format (JPEG, PNG, WEBP).
            quality: JPEG/WEBP quality (1-100).

        Returns:
            Base64-encoded string of the image.
        """
        buffer = io.BytesIO()
        save_kwargs: dict = {"format": fmt}
        if fmt.upper() in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
        if img.mode == "RGBA" and fmt.upper() == "JPEG":
            img = img.convert("RGB")
        img.save(buffer, **save_kwargs)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def pil_to_numpy(img: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array (RGB)."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def numpy_to_pil(arr: np.ndarray) -> Image.Image:
        """Convert numpy array (RGB) to PIL Image."""
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def save_to_disk(img: Image.Image, path: Path, fmt: str = "JPEG") -> Path:
        """Save a PIL image to disk and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict = {"format": fmt}
        if fmt.upper() == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(str(path), **save_kwargs)
        logger.info("Saved image to %s", path)
        return path
