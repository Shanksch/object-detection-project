"""
Configuration Settings for Object Detection Application
======================================================

This file contains all configuration parameters for the object detection app.
Modify these values according to your system setup and requirements.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# =====================
# SYSTEM CONFIGURATION
# =====================

# Auto-detect Python interpreter path for better portability
def get_python_path() -> str:
    """Get the current Python interpreter path"""
    return sys.executable

PYTHON_PATH = get_python_path()

# =====================
# MODEL CONFIGURATION
# =====================

# YOLOv5 model weights file
YOLO_WEIGHTS = "best.pt"

# Input image size (square dimension)
IMAGE_SIZE = 416

# Detection confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# IoU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# Maximum number of detections per image
MAX_DETECTIONS = 1000

# =====================
# APPLICATION SETTINGS
# =====================

# Flask server configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8080"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# File paths
DATA_DIR = "data"
UPLOAD_FILENAME = "inputImage.jpg"
RESULTS_DIR = "yolov5/runs/detect"

# Security settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
TEMP_FOLDER = os.path.join(DATA_DIR, "temp")

# Performance settings
MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "1"))
MAX_CONCURRENT_DETECTIONS = int(os.getenv("MAX_CONCURRENT_DETECTIONS", "4"))
DETECTION_TIMEOUT = int(os.getenv("DETECTION_TIMEOUT", "30"))  # seconds

# =====================
# DETECTION CLASSES
# =====================

# Object classes that the model can detect
DETECTION_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"
    # Add more classes based on your trained model
]

# =====================
# LOGGING CONFIGURATION
# =====================

# Enable/disable console logging
ENABLE_LOGGING = True

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# =====================
# HELPER FUNCTIONS
# =====================

def get_upload_path():
    """Get the full path for uploaded images"""
    return os.path.join(DATA_DIR, UPLOAD_FILENAME)

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if Python path exists
    if not os.path.exists(PYTHON_PATH):
        errors.append(f"Python interpreter not found at: {PYTHON_PATH}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
        except Exception as e:
            errors.append(f"Cannot create data directory: {e}")
    
    # Create upload and temp directories
    for directory in [UPLOAD_FOLDER, TEMP_FOLDER]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {directory}: {e}")
    
    # Validate thresholds
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    if not 0.0 <= IOU_THRESHOLD <= 1.0:
        errors.append("IOU_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate file size limits
    if MAX_FILE_SIZE <= 0:
        errors.append("MAX_FILE_SIZE must be positive")
    
    # Validate performance settings
    if MODEL_CACHE_SIZE < 0:
        errors.append("MODEL_CACHE_SIZE must be non-negative")
    
    if MAX_CONCURRENT_DETECTIONS <= 0:
        errors.append("MAX_CONCURRENT_DETECTIONS must be positive")
    
    return errors

if __name__ == "__main__":
    # Validate configuration when run directly
    errors = validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ Configuration is valid!")
        print(f"   • Python Path: {PYTHON_PATH}")
        print(f"   • Model Weights: {YOLO_WEIGHTS}")
        print(f"   • Image Size: {IMAGE_SIZE}")
        print(f"   • Confidence: {CONFIDENCE_THRESHOLD}")
        print(f"   • Server: {FLASK_HOST}:{FLASK_PORT}")
