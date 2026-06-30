# 🎯 Object Detection Web Application

A FastAPI-based web application for real-time object detection using YOLOv5. Upload images or use the REST API to detect objects with bounding boxes and confidence scores.

**Live Demo**: [https://shasha-nks-object-detection-studio.hf.space](https://shasha-nks-object-detection-studio.hf.space)

## 📋 Features

- **Image Upload Detection**: Upload any image and get object detection results
- **Live Camera Detection**: Real-time object detection from webcam
- **Web Interface**: User-friendly HTML interface for easy testing
- **REST API**: JSON API endpoints for integration
- **Model Training**: Built-in pipeline for training custom models
- **Dockerized**: Fully containerized for cloud deployment (Hugging Face Spaces)

## 🎯 Detected Objects

The trained model is based on the COCO128 dataset and can detect:
- **Person** 👤
- **Bicycle** 🚲  
- **Car** 🚗
- **Motorcycle** 🏍️
- **Airplane** ✈️
- **Bus** 🚌
- **Train** 🚂

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git LFS (to pull the model weights from a separate source)

### 1. Clone & Setup
```bash
git clone https://github.com/Shanksch/object-detection-project.git
cd object-detection-project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Model Weights
The YOLOv5 model weights (`best.pt`) are not included in the repo (too large for Git).
You need to place them manually:
```
yolov5/
├── best.pt          # Your trained model weights
└── ...              # YOLOv5 framework files
```

### 3. Run the Application
```bash
# Option A: Use the batch script (Windows)
.\run_app.bat

# Option B: Run directly
python -m app.main
```

### 4. Access the Web Interface
Open your browser and navigate to:
- **Main App**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📂 Project Structure

```
object-detection-project/
├── app/                      # FastAPI application (production)
│   ├── main.py               # FastAPI entry point
│   ├── config/               # Pydantic settings (.env based)
│   ├── api/routes/           # API route handlers
│   │   ├── detection.py      # /predict, /predict/file endpoints
│   │   ├── health.py         # /health endpoint
│   │   └── training.py       # /train endpoint
│   ├── services/             # Business logic
│   │   ├── model_manager.py  # YOLOv5 model loading & inference
│   │   ├── detection_service.py
│   │   ├── image_service.py
│   │   └── training_service.py
│   ├── core/                 # Exceptions, logging
│   └── ml/pipeline/          # ML training pipeline
│
├── objectDetection/          # Legacy ML pipeline
│   ├── components/           # Data ingestion, validation, training
│   ├── pipeline/             # Training pipeline orchestration
│   ├── logger/               # Custom logging
│   └── utils/                # Utility functions
│
├── templates/
│   └── modern_ui.html        # Main web interface
│
├── static/                   # Static assets
├── Dockerfile                # Docker configuration
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── run_app.bat               # Windows startup script
```

> **Note**: The following directories are local-only and not tracked in Git:
> `yolov5/`, `datasets/`, `data/`, `logs/`, `artifacts/`, `object-detection-studio/`

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API status, model info, uptime |
| `POST` | `/api/v1/predict` | Detect objects in a base64-encoded image |
| `POST` | `/api/v1/predict/file` | Detect objects in an uploaded image file |
| `GET` | `/api/v1/live` | Live camera detection status |
| `POST` | `/api/v1/train` | Start model training (background task) |
| `GET` | `/api/v1/train/{task_id}` | Poll training progress |
| `GET` | `/docs` | Interactive Swagger API docs |

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model
MODEL_WEIGHTS=best.pt
IMAGE_SIZE=416
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
DEVICE=auto
```

## 🐳 Docker Deployment

```bash
docker build -t object-detection-studio .
docker run -d -p 7860:7860 object-detection-studio
```

The app is also deployed on [Hugging Face Spaces](https://huggingface.co/spaces/shasha-nks/object-detection-studio).

## 📊 Performance

- **Detection Speed**: ~40-60ms per image (CPU)
- **Model Size**: 7.2M parameters (YOLOv5s)
- **Input Size**: 416x416 pixels (configurable)

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Shashank Chauhan**
- Email: shashankchauhan8577@gmail.com
