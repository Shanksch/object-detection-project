"""
Model Trainer Pipeline
=======================
Trains YOLOv5 model using subprocess.run() with proper error handling.
Replaces the dangerous os.system() calls in the original code.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from app.config.settings import Settings, get_settings
from app.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class ModelTrainerPipeline:
    """Handles YOLOv5 model training with safe subprocess execution."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._yolo_dir = self._settings.base_dir / self._settings.yolo_dir

    def run(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute YOLOv5 training pipeline.

        Args:
            epochs: Number of training epochs.
            batch_size: Training batch size.
            image_size: Training image size.

        Returns:
            Dict with 'model_path' and training metadata.

        Raises:
            TrainingError: If any step fails.
        """
        epochs = epochs or self._settings.train_epochs
        batch_size = batch_size or self._settings.train_batch_size
        image_size = image_size or self._settings.image_size

        # Step 1: Prepare dataset
        dataset_dir = self._settings.base_dir / "object_detection_dataset"
        data_yaml = dataset_dir / "data.yaml"

        if not data_yaml.exists():
            # Try to extract from zip if available
            zip_path = self._settings.base_dir / "object_detection_dataset.zip"
            if zip_path.exists():
                self._extract_dataset(zip_path)
            else:
                raise TrainingError(
                    "Dataset not found. Run data ingestion first.",
                    detail={"expected_path": str(data_yaml)},
                )

        # Step 2: Read number of classes from data.yaml
        num_classes = self._read_num_classes(data_yaml)
        logger.info("Training with %d classes, %d epochs, batch %d", num_classes, epochs, batch_size)

        # Step 3: Create custom model config
        self._create_custom_config(num_classes)

        # Step 4: Run YOLOv5 training
        self._run_training(epochs, batch_size, image_size, data_yaml)

        # Step 5: Copy best weights
        model_path = self._copy_best_weights()

        # Step 6: Cleanup
        self._cleanup()

        logger.info("✅ Training complete. Best model saved to: %s", model_path)
        return {
            "model_path": str(model_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "num_classes": num_classes,
        }

    def _extract_dataset(self, zip_path: Path) -> None:
        """Extract dataset ZIP file."""
        import zipfile
        logger.info("Extracting dataset from %s", zip_path)
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(self._settings.base_dir))
        except Exception as exc:
            raise TrainingError(f"Failed to extract dataset: {exc}") from exc

    def _read_num_classes(self, data_yaml: Path) -> int:
        """Read the number of classes from data.yaml."""
        try:
            with open(data_yaml, "r") as f:
                config = yaml.safe_load(f)
            nc = config.get("nc")
            if nc is None:
                raise TrainingError("data.yaml missing 'nc' field")
            return int(nc)
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"Failed to read data.yaml: {exc}") from exc

    def _create_custom_config(self, num_classes: int) -> None:
        """Create a custom YOLOv5 model config with the correct number of classes."""
        weight_name = self._settings.pretrained_weights.split(".")[0]
        source_config = self._yolo_dir / "models" / f"{weight_name}.yaml"
        custom_config = self._yolo_dir / "models" / f"custom_{weight_name}.yaml"

        if not source_config.exists():
            raise TrainingError(
                f"Base model config not found: {source_config}",
                detail={"expected": str(source_config)},
            )

        try:
            with open(source_config, "r") as f:
                config = yaml.safe_load(f)
            config["nc"] = num_classes
            with open(custom_config, "w") as f:
                yaml.dump(config, f)
            logger.info("Created custom config: %s (nc=%d)", custom_config, num_classes)
        except Exception as exc:
            raise TrainingError(f"Failed to create custom config: {exc}") from exc

    def _run_training(
        self,
        epochs: int,
        batch_size: int,
        image_size: int,
        data_yaml: Path,
    ) -> None:
        """Run YOLOv5 training via subprocess (safe replacement for os.system)."""
        weight_name = self._settings.pretrained_weights.split(".")[0]
        cmd = [
            sys.executable,
            "train.py",
            "--img", str(image_size),
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", str(data_yaml),
            "--cfg", f"./models/custom_{weight_name}.yaml",
            "--weights", self._settings.pretrained_weights,
            "--name", "yolov5s_results",
            "--cache",
        ]

        logger.info("Starting training: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._yolo_dir),
                capture_output=True,
                text=True,
                timeout=3600 * 12,  # 12 hour timeout
            )

            if result.returncode != 0:
                logger.error("Training stderr:\n%s", result.stderr[-2000:] if result.stderr else "N/A")
                raise TrainingError(
                    f"YOLOv5 training failed with return code {result.returncode}",
                    detail={"stderr": result.stderr[-500:] if result.stderr else None},
                )

            logger.info("Training stdout (last 500 chars):\n%s", result.stdout[-500:] if result.stdout else "N/A")

        except subprocess.TimeoutExpired:
            raise TrainingError("Training timed out after 12 hours")
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"Failed to execute training: {exc}") from exc

    def _copy_best_weights(self) -> Path:
        """Copy best.pt to the expected locations."""
        source = self._yolo_dir / "runs" / "train" / "yolov5s_results" / "weights" / "best.pt"
        if not source.exists():
            raise TrainingError(
                "Training completed but best.pt not found",
                detail={"expected": str(source)},
            )

        # Copy to yolo dir root
        dest_yolo = self._yolo_dir / "best.pt"
        shutil.copy2(str(source), str(dest_yolo))

        # Copy to artifacts
        timestamp_dir = self._settings.base_dir / self._settings.artifacts_dir / "model_trainer"
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        dest_artifacts = timestamp_dir / "best.pt"
        shutil.copy2(str(source), str(dest_artifacts))

        return dest_yolo

    def _cleanup(self) -> None:
        """Clean up temporary training files."""
        cleanup_paths = [
            self._yolo_dir / "runs",
            self._settings.base_dir / "train",
            self._settings.base_dir / "test",
            self._settings.base_dir / "data.yaml",
        ]
        for path in cleanup_paths:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(str(path), ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                logger.debug("Cleaned up: %s", path)
