"""
Configuration Settings for Object Detection Application
======================================================

This file contains all configuration parameters for the object detection app.
Modify these values according to your system setup and requirements.
"""

import os

# =====================
# SYSTEM CONFIGURATION
# =====================

# Python interpreter path (update this to match your environment)
PYTHON_PATH = "C:\\Users\\shash\\anaconda3\\envs\\objectdetect\\python.exe"

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
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
FLASK_DEBUG = False

# File paths
DATA_DIR = "data"
UPLOAD_FILENAME = "inputImage.jpg"
RESULTS_DIR = "yolov5/runs/detect"

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
    
    # Validate thresholds
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    if not 0.0 <= IOU_THRESHOLD <= 1.0:
        errors.append("IOU_THRESHOLD must be between 0.0 and 1.0")
    
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
