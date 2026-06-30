"""
Object Detection Studio — FastAPI Entry Point
===============================================
Production-grade FastAPI application with:
- Lifespan context manager for model preloading
- Structured exception handling
- CORS configuration
- Auto-generated OpenAPI documentation at /docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import get_settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup:
      - Configure logging
      - Load YOLOv5 model into memory
      - Run warm-up inference

    Shutdown:
      - Unload model and free GPU memory
    """
    settings = get_settings()

    # Setup structured logging
    log_dir = settings.base_dir / settings.log_dir if not settings.debug else None
    setup_logging(
        level=settings.log_level,
        log_dir=log_dir,
        json_format=settings.log_json,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )

    logger.info("=" * 60)
    logger.info("  🎯 Object Detection Studio — Starting Up")
    logger.info("=" * 60)
    logger.info("  Device: %s", settings.resolve_device())
    logger.info("  Model:  %s", settings.model_weights)
    logger.info("  Server: %s:%d", settings.host, settings.port)
    logger.info("=" * 60)

    # Load model at startup
    from app.services.model_manager import ModelManager
    model_manager = ModelManager.get_instance(settings)
    try:
        model_manager.load_model()
        model_manager.warm_up(image_size=settings.image_size)
    except Exception as exc:
        logger.warning("⚠️ Model loading failed (app will start in degraded mode): %s", exc)

    yield  # ← Application runs here

    # Shutdown
    logger.info("Shutting down — unloading model...")
    model_manager.unload()
    logger.info("👋 Goodbye!")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Object Detection Studio",
        description=(
            "Production-grade object detection API powered by YOLOv5. "
            "Upload images or use live camera for real-time detection."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ──────────────────────────────────────────
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.error("AppException: %s (status=%d)", exc.message, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": True, "message": "Endpoint not found", "status_code": 404},
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": "Internal server error", "status_code": 500},
        )

    # ── Request timing middleware ───────────────────────────────────
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        if elapsed > 100:  # Only log slow requests
            logger.info("%s %s — %.1fms", request.method, request.url.path, elapsed)
        return response

    # ── Register routes ─────────────────────────────────────────────
    from app.api.routes.health import router as health_router
    from app.api.routes.detection import router as detection_router
    from app.api.routes.training import router as training_router

    app.include_router(health_router)
    app.include_router(detection_router)
    app.include_router(training_router)

    # ── Static files ────────────────────────────────────────────────
    static_dir = settings.base_dir / settings.static_dir
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import sys
    import uvicorn

    # Force UTF-8 output on Windows to handle unicode characters
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()

    print("=" * 55)
    print("  [*] Object Detection Studio")
    print("=" * 55)
    print(f"  [>] Main App:  http://localhost:{settings.port}/")
    print(f"  [>] API Docs:  http://localhost:{settings.port}/docs")
    print(f"  [>] Health:    http://localhost:{settings.port}/health")
    print(f"  [>] Predict:   http://localhost:{settings.port}/api/v1/predict")
    print(f"  [>] Upload:    http://localhost:{settings.port}/api/v1/predict/file")
    print(f"  [>] Train:     http://localhost:{settings.port}/api/v1/train")
    print(f"  [>] Live:      http://localhost:{settings.port}/api/v1/live")
    print("=" * 55)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
    )
