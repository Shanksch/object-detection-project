"""
Training Routes
================
Endpoints for model training with async execution and status polling.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.schemas import TrainRequest, TrainResponse, TrainStatusResponse
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Training"])

# Singleton training service
_training_service: Optional[TrainingService] = None


def _get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Start model training",
    description="Starts a training pipeline as a background task. Returns a task_id for polling status.",
)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Start model training as a background task.

    Returns a task_id that can be used to poll for status at GET /api/v1/train/{task_id}.
    """
    service = _get_training_service()

    task = service.create_task(
        epochs=request.epochs,
        batch_size=request.batch_size,
        image_size=request.image_size,
    )

    # Run training in the background (non-blocking)
    background_tasks.add_task(service.run_training, task.task_id)

    logger.info("Training task %s queued", task.task_id)

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "message": f"Training pipeline started. Poll status at /api/v1/train/{task.task_id}",
    }


@router.get(
    "/train/{task_id}",
    response_model=TrainStatusResponse,
    summary="Get training status",
    description="Poll the status of a running or completed training task.",
)
async def get_training_status(task_id: str) -> dict:
    """
    Get the status of a training task by its ID.

    Possible status values: pending, ingesting, validating, training, completed, failed.
    """
    service = _get_training_service()
    task = service.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Training task not found: {task_id}")

    return task.to_dict()


@router.get(
    "/train",
    summary="List all training tasks",
    description="Returns a list of all training tasks (active and completed).",
)
async def list_training_tasks() -> dict:
    """List all training tasks."""
    service = _get_training_service()
    return {"tasks": service.list_tasks()}
