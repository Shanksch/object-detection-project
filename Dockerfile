# ============================================================
#  Object Detection Studio — Production Dockerfile
#  Multi-stage build  •  Python 3.11-slim  •  FastAPI + YOLOv5
# ============================================================
#
#  Build:   docker build -t object-detection-studio .
#  Run:     docker run -d -p 7860:7860 --env-file .env object-detection-studio
#
#  NOTE:    The yolov5/ directory (with best.pt) must be present in the
#           build context. It is gitignored, so after a fresh clone you
#           need to place it manually before building.
# ============================================================


# ── Stage 1: Install Python dependencies ───────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

# Only the compiler toolchain needed during pip install
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Collect only the YOLOv5 files we need ─────────
FROM python:3.11-slim AS yolo-filter

WORKDIR /yolo

# Copy the full repo, then delete what isn't needed at runtime
COPY yolov5/ .
RUN rm -rf .git __pycache__ runs classify segment \
           tutorial.ipynb CONTRIBUTING.md README*.md \
           .gitignore .gitattributes .dockerignore \
           yolov5s.pt best.torchscript \
           pyproject.toml requirements.txt


# ── Stage 3: Final runtime image ──────────────────────────
FROM python:3.11-slim AS runtime

# OCI metadata
LABEL org.opencontainers.image.title="Object Detection Studio" \
      org.opencontainers.image.description="FastAPI + YOLOv5 real-time object detection API" \
      org.opencontainers.image.authors="Shashank Chauhan <shashankchauhan8577@gmail.com>" \
      org.opencontainers.image.source="https://github.com/Shanksch/object-detection-project"

# Minimal runtime system deps (OpenCV needs libgl1, ffmpeg for video)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ffmpeg libsm6 libxext6 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python packages from stage 1 ──────────────────────────
COPY --from=deps /install /usr/local

# ── Application layers (ordered by change frequency) ──────
#    Rarely-changing layers first → better Docker cache hits.

# 1. Default env config
COPY .env.example .env

# 2. YOLOv5 core + trained model weights (changes rarely)
COPY --from=yolo-filter /yolo yolov5/

# 3. Frontend assets
COPY static/    static/
COPY templates/ templates/

# 4. Application code (changes most often → last layer)
COPY app/ app/

# ── Runtime directories ───────────────────────────────────
RUN mkdir -p data artifacts logs

# ── Non-root user (CIS Docker Benchmark §4.1) ─────────────
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Environment ───────────────────────────────────────────
EXPOSE 7860

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

# ── Health check ──────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entrypoint ────────────────────────────────────────────
CMD ["python", "-m", "app.main"]