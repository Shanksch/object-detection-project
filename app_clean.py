#!/usr/bin/env python3
"""
Object Detection Web Application
===============================

A Flask-based web application for real-time object detection using YOLOv5.
Supports image upload, live camera feed, and model training.

Author: Shashank Chauhan
Dependencies: Flask, YOLOv5, OpenCV, PyTorch
"""

import sys
import os
import glob
import shutil
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

# Import custom modules
from objectDetection.exception import SignException
from objectDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from objectDetection.pipeline.training_pipeline import TrainPipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
PYTHON_PATH = "C:\\Users\\shash\\anaconda3\\envs\\objectdetect\\python.exe"
YOLO_WEIGHTS = "best.pt"
IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5

class ObjectDetectionApp:
    """Main application class for object detection"""
    
    def __init__(self):
        self.input_filename = "inputImage.jpg"
        self.input_path = f"data/{self.input_filename}"
    
    def run_detection(self, source_path):
        """
        Run YOLOv5 object detection on the specified source
        
        Args:
            source_path (str): Path to input image or camera index
            
        Returns:
            int: System command exit code
        """
        cmd = (f'cd yolov5 & {PYTHON_PATH} detect.py '
               f'--weights {YOLO_WEIGHTS} --img {IMAGE_SIZE} '
               f'--conf {CONFIDENCE_THRESHOLD} --source {source_path}')
        return os.system(cmd)
    
    def get_latest_detection_result(self):
        """
        Find and return the path to the latest detection result image
        
        Returns:
            str or None: Path to result image or None if not found
        """
        exp_dirs = glob.glob("yolov5/runs/detect/exp*")
        if not exp_dirs:
            return None
            
        latest_exp = max(exp_dirs, key=os.path.getctime)
        result_path = os.path.join(latest_exp, self.input_filename)
        result_path = result_path.replace('/', '\\')  # Windows path fix
        
        return result_path if os.path.exists(result_path) else None
    
    def cleanup_results(self):
        """Clean up temporary detection results"""
        if os.path.exists("yolov5/runs"):
            shutil.rmtree("yolov5/runs")

# Initialize app instance
detection_app = ObjectDetectionApp()

# =====================
# WEB ROUTES
# =====================

@app.route("/")
def home():
    """Modern, beautiful UI for object detection - Main application page"""
    return render_template("modern_ui.html")

@app.route("/health")
def health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Object Detection API is running!",
        "model": YOLO_WEIGHTS,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    """
    Object detection endpoint for image upload
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "image": "base64_encoded_result_image"
    } or {"error": "error_message"}
    """
    try:
        # Validate request
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "Missing image data in request"}), 400
        
        # Decode and save input image
        image_data = request.json['image']
        decodeImage(image_data, detection_app.input_filename)
        
        print(f"🔍 Running object detection on uploaded image...")
        
        # Run detection
        result_code = detection_app.run_detection(f"../{detection_app.input_path}")
        
        if result_code != 0:
            return jsonify({"error": "Detection process failed"}), 500
        
        # Get result image
        result_path = detection_app.get_latest_detection_result()
        
        if result_path:
            # Encode result image
            encoded_image = encodeImageIntoBase64(result_path)
            result = {"image": encoded_image.decode('utf-8')}
            print("✅ Object detection completed successfully!")
        else:
            result = {"error": "Detection result image not found"}
            print("❌ Detection result image not found")
        
        # Cleanup
        detection_app.cleanup_results()
        
        return jsonify(result)
        
    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400
        
    except Exception as e:
        error_msg = f"Detection failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/live", methods=['GET'])
@cross_origin()
def live_detection():
    """
    Live camera detection status endpoint
    The actual camera access is handled by the frontend using WebRTC
    """
    try:
        # Just return a success message - the frontend handles camera access
        return jsonify({
            "status": "ready",
            "message": "Live camera detection is ready. Camera access is handled by the browser.",
            "instructions": "Use the 'Capture & Analyze' button to take snapshots for detection."
        })
            
    except Exception as e:
        error_msg = f"Live detection status failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/train", methods=['GET'])
def train_model():
    """
    Start model training pipeline
    Warning: This is a time-intensive operation
    """
    try:
        print("🏋️ Starting model training...")
        trainer = TrainPipeline()
        trainer.run_pipeline()
        return jsonify({"message": "Training completed successfully!"})
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

# =====================
# ERROR HANDLERS
# =====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# =====================
# MAIN APPLICATION
# =====================

if __name__ == "__main__":
    print("=" * 50)
    print("    🎯 Object Detection Web Application")
    print("=" * 50)
    print()
    print("📋 Available endpoints:")
    print("   🏠 Main App:      http://localhost:8080/ (Modern UI)")
    print("   💚 Health Check:  http://localhost:8080/health")
    print("   🎯 Prediction:    http://localhost:8080/predict (POST)")
    print("   📹 Live Camera:   http://localhost:8080/live (GET)")
    print("   🏋️  Training:      http://localhost:8080/train (GET)")
    print()
    print("🚀 Starting Flask server...")
    print("   Press Ctrl+C to stop")
    print()
    
    app.run(host="0.0.0.0", port=8080, debug=False)
