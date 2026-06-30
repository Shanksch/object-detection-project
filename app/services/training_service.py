"""
Training Service
=================
Orchestrates the training pipeline with status tracking.
Runs training as a background task with progress polling.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.config.settings import Settings, get_settings
from app.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    VALIDATING = "validating"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingTask:
    """Tracks a single training run."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    message: str = "Waiting to start"
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": round(
                ((self.completed_at or datetime.now(timezone.utc)) - self.started_at).total_seconds(), 1
            ) if self.started_at else None,
            "config": self.config,
        }


class TrainingService:
    """Manages training pipeline execution with status tracking."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._tasks: dict[str, TrainingTask] = {}

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """Retrieve a training task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict[str, Any]]:
        """List all training tasks."""
        return [t.to_dict() for t in self._tasks.values()]

    def create_task(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
    ) -> TrainingTask:
        """Create a new training task (does not start it)."""
        task = TrainingTask(
            config={
                "epochs": epochs or self._settings.train_epochs,
                "batch_size": batch_size or self._settings.train_batch_size,
                "image_size": image_size or self._settings.image_size,
                "weights": self._settings.pretrained_weights,
                "data_url": self._settings.data_download_url,
            }
        )
        self._tasks[task.task_id] = task
        logger.info("Created training task %s", task.task_id)
        return task

    async def run_training(self, task_id: str) -> None:
        """
        Execute the full training pipeline for a given task.

        Stages: Data Ingestion → Data Validation → Model Training

        This is designed to be called from FastAPI's BackgroundTasks.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise TrainingError(f"Training task not found: {task_id}")

        task.started_at = datetime.now(timezone.utc)

        try:
            # Stage 1: Data Ingestion
            task.status = TrainingStatus.INGESTING
            task.message = "Downloading dataset..."
            task.progress = 10.0
            logger.info("[%s] Stage 1: Data ingestion", task_id)

            from app.ml.pipeline.data_ingestion import DataIngestionPipeline
            ingestion = DataIngestionPipeline(self._settings)
            ingestion_result = await ingestion.run(task.config["data_url"])

            task.progress = 30.0

            # Stage 2: Data Validation
            task.status = TrainingStatus.VALIDATING
            task.message = "Validating dataset..."
            task.progress = 40.0
            logger.info("[%s] Stage 2: Data validation", task_id)

            from app.ml.pipeline.data_validation import DataValidationPipeline
            validation = DataValidationPipeline(self._settings)
            validation.run(ingestion_result["feature_store_path"])

            task.progress = 50.0

            # Stage 3: Model Training
            task.status = TrainingStatus.TRAINING
            task.message = f"Training model ({task.config['epochs']} epochs)..."
            task.progress = 60.0
            logger.info("[%s] Stage 3: Model training", task_id)

            from app.ml.pipeline.model_trainer import ModelTrainerPipeline
            trainer = ModelTrainerPipeline(self._settings)
            trainer.run(
                epochs=task.config["epochs"],
                batch_size=task.config["batch_size"],
                image_size=task.config["image_size"],
            )

            # Done
            task.status = TrainingStatus.COMPLETED
            task.message = "Training completed successfully"
            task.progress = 100.0
            task.completed_at = datetime.now(timezone.utc)
            logger.info("✅ [%s] Training pipeline completed", task_id)

        except Exception as exc:
            task.status = TrainingStatus.FAILED
            task.error = str(exc)
            task.message = f"Training failed: {exc}"
            task.completed_at = datetime.now(timezone.utc)
            logger.error("❌ [%s] Training failed: %s", task_id, exc, exc_info=True)
