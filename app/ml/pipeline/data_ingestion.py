"""
Data Ingestion Pipeline
========================
Downloads and extracts the training dataset.
Replaces the legacy DataIngestion class with async HTTP download,
progress tracking, and path traversal protection.
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from app.config.settings import Settings, get_settings
from app.core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Handles dataset download and extraction."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    async def run(self, url: Optional[str] = None) -> dict[str, Any]:
        """
        Execute the data ingestion pipeline.

        Args:
            url: Dataset download URL. Defaults to settings.

        Returns:
            Dict with 'zip_path' and 'feature_store_path'.

        Raises:
            DataIngestionError: If download or extraction fails.
        """
        download_url = url or self._settings.data_download_url
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        # Setup directories
        artifacts_dir = self._settings.base_dir / self._settings.artifacts_dir / timestamp
        ingestion_dir = artifacts_dir / "data_ingestion"
        feature_store_dir = ingestion_dir / "feature_store"

        ingestion_dir.mkdir(parents=True, exist_ok=True)
        feature_store_dir.mkdir(parents=True, exist_ok=True)

        # Download
        zip_path = await self._download(download_url, ingestion_dir)

        # Extract
        self._extract(zip_path, feature_store_dir)

        result = {
            "zip_path": str(zip_path),
            "feature_store_path": str(feature_store_dir),
            "artifacts_dir": str(artifacts_dir),
        }
        logger.info("Data ingestion completed: %s", result)
        return result

    async def _download(self, url: str, dest_dir: Path) -> Path:
        """Download dataset from URL."""
        filename = url.split("/")[-1]
        zip_path = dest_dir / filename

        logger.info("Downloading dataset from %s ...", url)

        try:
            # Use httpx if available (async), fall back to urllib
            try:
                import httpx
                async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    zip_path.write_bytes(response.content)
            except ImportError:
                # Fallback to synchronous urllib
                import urllib.request
                urllib.request.urlretrieve(url, str(zip_path))

            logger.info("Downloaded %s (%.1f MB)", filename, zip_path.stat().st_size / 1e6)
            return zip_path

        except Exception as exc:
            raise DataIngestionError(
                f"Failed to download dataset from {url}: {exc}",
                detail={"url": url},
            ) from exc

    def _extract(self, zip_path: Path, dest_dir: Path) -> None:
        """Extract ZIP file with path traversal protection."""
        if not zip_path.exists():
            raise DataIngestionError(f"ZIP file not found: {zip_path}")

        logger.info("Extracting %s to %s ...", zip_path.name, dest_dir)

        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                # Path traversal check
                for member in zf.namelist():
                    member_path = (dest_dir / member).resolve()
                    if not str(member_path).startswith(str(dest_dir.resolve())):
                        raise DataIngestionError(
                            f"ZIP path traversal detected: {member}",
                            detail={"malicious_path": member},
                        )
                zf.extractall(str(dest_dir))

            logger.info("Extraction complete: %d files", len(os.listdir(dest_dir)))

        except zipfile.BadZipFile:
            raise DataIngestionError(f"Corrupted ZIP file: {zip_path}")
        except DataIngestionError:
            raise
        except Exception as exc:
            raise DataIngestionError(f"Extraction failed: {exc}") from exc
