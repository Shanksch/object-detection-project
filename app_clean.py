from yolov5 import detect
import sys
import os
import glob
import shutil
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# Safe PyTorch globals for YOLOv5 model load
import torch
try:
    from yolov5 import models
    torch.serialization.add_safe_globals([models.yolo.DetectionModel])
except Exception:
    pass  # if not in same process as YOLOv5, safe to ignore

# Import custom modules
from objectDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from objectDetection.pipeline.training_pipeline import TrainPipeline

# =====================
# CONFIG
# =====================
PYTHON_PATH = os.getenv("PYTHON_PATH", sys.executable)
YOLO_DIR = "yolov5"
YOLO_WEIGHTS = "best.pt"
IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5

# =====================
# FLASK APP
# =====================
app = Flask(__name__)
CORS(app)

class ObjectDetectionApp:
    """Main application class for object detection."""

    def __init__(self):
        self.input_filename = "inputImage.jpg"
        self.input_path = os.path.join("data", self.input_filename)

    def run_detection(self, source_path):
        """
        Run YOLOv5 object detection on the specified source.
        Returns True if successful, False otherwise.
        """
        cmd = [
            PYTHON_PATH,
            "detect.py",
            "--weights", YOLO_WEIGHTS,
            "--img", str(IMAGE_SIZE),
            "--conf", str(CONFIDENCE_THRESHOLD),
            "--source", source_path
        ]
        result = subprocess.run(cmd, cwd=YOLO_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Detection failed:")
            print(result.stderr)
            return False
        print("✅ Detection succeeded.")
        return True

    def get_latest_detection_result(self):
        """
        Find and return the path to the latest detection result image.
        """
        exp_dirs = glob.glob(os.path.join(YOLO_DIR, "runs", "detect", "exp*"))
        if not exp_dirs:
            return None
        latest_exp = max(exp_dirs, key=os.path.getctime)
        result_path = os.path.join(latest_exp, self.input_filename)
        return result_path if os.path.exists(result_path) else None

    def cleanup_results(self):
        """Clean up temporary detection results."""
        runs_dir = os.path.join(YOLO_DIR, "runs")
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir, ignore_errors=True)


# Initialize app instance
detection_app = ObjectDetectionApp()

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    """Main UI page."""
    return render_template("modern_ui.html")


@app.route("/health")
def health_check():
    """Health check endpoint."""
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
    Object detection endpoint for image upload.
    """
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "Missing image data in request"}), 400

        # Decode and save input image
        image_data = request.json['image']
        decodeImage(image_data, detection_app.input_filename)
        print("🔍 Running object detection on uploaded image...")

        if not detection_app.run_detection(f"../{detection_app.input_path}"):
            return jsonify({"error": "Detection process failed"}), 500

        # Get result image
        result_path = detection_app.get_latest_detection_result()
        if result_path:
            encoded_image = encodeImageIntoBase64(result_path)
            response = {"image": encoded_image.decode('utf-8')}
            print("✅ Object detection completed successfully!")
        else:
            response = {"error": "Detection result image not found"}
            print("❌ Detection result image not found")

        detection_app.cleanup_results()
        return jsonify(response)

    except Exception as e:
        error_msg = f"Detection failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route("/live", methods=['GET'])
@cross_origin()
def live_detection():
    """
    Live camera detection status endpoint.
    """
    return jsonify({
        "status": "ready",
        "message": "Live camera detection is ready. Camera access is handled by the browser.",
        "instructions": "Use the 'Capture & Analyze' button to take snapshots for detection."
    })


@app.route("/train", methods=['GET'])
def train_model():
    """
    Start model training pipeline.
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
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error"}), 500


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print("=" * 50)
    print(" 🎯 Object Detection Web Application")
    print("=" * 50)
    print(" 🏠 Main App:      http://localhost:8080/")
    print(" 💚 Health Check:  http://localhost:8080/health")
    print(" 🎯 Prediction:    http://localhost:8080/predict (POST)")
    print(" 📹 Live Camera:   http://localhost:8080/live (GET)")
    print(" 🏋️ Training:      http://localhost:8080/train (GET)")
    print("\n🚀 Starting Flask server... (Press Ctrl+C to stop)\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
