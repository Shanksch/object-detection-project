# 🎯 Object Detection Web Application

A Flask-based web application for real-time object detection using YOLOv5. Upload images or use live camera feed to detect objects with bounding boxes and confidence scores.

## 📋 Features

- **Image Upload Detection**: Upload any image and get object detection results
- **Live Camera Detection**: Real-time object detection from webcam
- **Web Interface**: User-friendly HTML interface for easy testing
- **REST API**: JSON API endpoints for integration
- **Model Training**: Built-in pipeline for training custom models
- **Multi-class Detection**: Supports person, vehicle, and other object categories

## 🎯 Detected Objects

The trained model can detect:
- **Person** 👤
- **Bicycle** 🚲  
- **Car** 🚗
- **Motorcycle** 🏍️
- **Airplane** ✈️
- **Bus** 🚌
- **Train** 🚂

## 🚀 Quick Start

### 1. Run the Application
```bash
# Start the server
.\run_app.bat
```

### 2. Access the Web Interface
Open your browser and navigate to:
- **Main App**: http://localhost:8080/
- **Health Check**: http://localhost:8080/health

### 3. Upload and Detect
1. Go to the test page
2. Click "Select Image" and choose a photo
3. Click "Detect Objects" to see results
4. View the original and result images with bounding boxes

## 📂 Project Structure

```
object-detection-project/
├── app.py           # Main Flask application (CLEANED)
├── config.py             # Configuration settings
├── run_app.bat           # Startup script
├── README.md       # This documentation
│
├── templates/
│   ├── index.html        # Main web interface
│   └── test.html         # Simple test interface
│
├── objectDetection/      # Custom detection pipeline
│   ├── components/       # Data processing components
│   ├── pipeline/         # Training and inference pipelines
│   ├── utils/            # Utility functions
│   └── exception/        # Custom exception handling
│
├── yolov5/              # YOLOv5 framework
│   ├── detect.py        # Detection script
│   ├── best.pt          # Trained model weights
│   └── ...              # Other YOLOv5 files
│
└── data/                # Input/output data
    └── inputImage.jpg   # Uploaded images
```

## 🌐 API Endpoints

### Health Check
```http
GET /health
```
Returns API status and configuration info.

### Image Detection
```http
POST /predict
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```
Returns detection results with bounding boxes.

### Live Camera Detection
```http
GET /live
```
Starts live camera detection (opens detection window).

### Model Training
```http
GET /train
```
Starts the model training pipeline (time-intensive).

## ⚙️ Configuration

Edit `config.py` to customize:
- Python environment path
- Model confidence threshold
- Image size and detection parameters
- Server host and port settings

## 🔧 System Requirements

- **Python**: 3.7+ (Conda environment recommended)
- **Dependencies**: Flask, YOLOv5, OpenCV, PyTorch
- **OS**: Windows (paths configured for Windows)
- **Memory**: 4GB+ RAM recommended
- **Camera**: Optional for live detection

## 📊 Performance

- **Detection Speed**: ~40-60ms per image
- **Model Size**: 7.2M parameters
- **Accuracy**: Custom trained on object detection dataset
- **Input Size**: 416x416 pixels (configurable)

## 🧪 Testing

Use the test page (`http://localhost:8080/test-page`) for easy testing:
1. **API Test Button**: Verifies server connection
2. **Image Upload**: Drag & drop or select files
3. **Detection Results**: Side-by-side original and detected images
4. **Status Messages**: Real-time feedback on operations

## 🐛 Troubleshooting

### Server Won't Start
- Check if Python path in `config.py` is correct
- Verify conda environment is activated
- Ensure port 8080 is available

### Detection Not Working
- Verify `best.pt` model file exists in `yolov5/` directory
- Check if input image is valid format (JPG, PNG)
- Review console logs for error messages

### No Results Displayed
- Ensure browser allows loading of base64 images
- Check browser developer console for JavaScript errors
- Try the test page instead of main interface

## 📝 Development

### Adding New Detection Classes
1. Retrain the model with new data
2. Update `DETECTION_CLASSES` in `config.py`
3. Replace `best.pt` with new model weights

### Modifying Detection Parameters
Edit `config.py`:
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (0.0-1.0)
- `IMAGE_SIZE`: Input image size (416, 640, etc.)
- `IOU_THRESHOLD`: Overlap threshold for duplicate removal

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Shashank Chauhan**
- Email: shashankchauhan8577@gmail.com
- Project: Object Detection Web Application

---

🎯 **Ready to detect objects!** Run `.\run_app.bat` to start the application.
