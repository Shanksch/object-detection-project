"""
Data Validation Pipeline
=========================
Validates the extracted dataset has all required files.

CRITICAL FIX: The original code had INVERTED validation logic.
It marked files NOT in the required list as "valid" (True).
This version correctly checks that ALL required files exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config.settings import Settings, get_settings
from app.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Required items in the dataset root
REQUIRED_ITEMS = {"train", "test", "data.yaml"}


class DataValidationPipeline:
    """Validates extracted dataset structure and content."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    def run(self, feature_store_path: str) -> bool:
        """
        Validate that the extracted dataset contains all required files.

        Args:
            feature_store_path: Path to the extracted dataset directory.

        Returns:
            True if validation passes.

        Raises:
            DataValidationError: If required files are missing.
        """
        store = Path(feature_store_path)

        if not store.exists():
            raise DataValidationError(
                f"Feature store directory not found: {store}",
                detail={"path": str(store)},
            )

        # List all items in the extracted directory
        # Handle case where dataset is nested inside a subdirectory
        items = {item.name for item in store.iterdir()}

        # Check if items are directly here or nested one level deep
        actual_root = store
        if not REQUIRED_ITEMS.issubset(items):
            # Check one level deep (common with ZIP extraction)
            subdirs = [d for d in store.iterdir() if d.is_dir()]
            for subdir in subdirs:
                sub_items = {item.name for item in subdir.iterdir()}
                if REQUIRED_ITEMS.issubset(sub_items):
                    actual_root = subdir
                    items = sub_items
                    break

        # Validate required items exist
        missing = REQUIRED_ITEMS - items
        if missing:
            raise DataValidationError(
                f"Dataset missing required items: {missing}",
                detail={
                    "missing": sorted(missing),
                    "found": sorted(items),
                    "path": str(actual_root),
                },
            )

        # Validate train/test directories are not empty
        for required_dir in ("train", "test"):
            dir_path = actual_root / required_dir
            if dir_path.is_dir():
                file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
                if file_count == 0:
                    raise DataValidationError(
                        f"Required directory '{required_dir}' is empty",
                        detail={"directory": str(dir_path)},
                    )
                logger.info("  ✓ %s/ contains %d files", required_dir, file_count)

        # Validate data.yaml is readable
        yaml_path = actual_root / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r") as f:
                    data_config = yaml.safe_load(f)
                if "nc" not in data_config:
                    logger.warning("data.yaml missing 'nc' (number of classes) field")
                else:
                    logger.info("  ✓ data.yaml: %d classes", data_config["nc"])
            except Exception as exc:
                raise DataValidationError(
                    f"data.yaml is not valid YAML: {exc}",
                    detail={"path": str(yaml_path)},
                ) from exc

        logger.info("✅ Dataset validation passed: %s", actual_root)
        return True
