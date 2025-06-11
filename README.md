# Car Retrieval System - Object Detection + Classification

This project implements a Car Retrieval System that combines YOLO object detection with Vision Transformer (ViT) classification to detect and classify Indonesian car types in videos.

## 🚗 Features

- **Object Detection**: YOLO model for detecting car instances in video frames
- **Car Classification**: Vision Transformer (ViT) model for classifying detected cars into 8 categories
- **Real-time Processing**: Process video files with detection and classification overlay
- **Indonesian Car Dataset**: Specifically trained on Indonesian car types

## 📋 Car Categories

The system can classify cars into 8 different types:
- Coupe
- Hatchback
- MPV (Multi-Purpose Vehicle)
- Pickup
- Sedan
- Sports Car
- SUV (Sport Utility Vehicle)
- Truck

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- At least 4GB of RAM

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd car-retrieval-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv car_detection_env

# Activate virtual environment
# On Windows:
car_detection_env\Scripts\activate

# On macOS/Linux:
source car_detection_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Download Required Model Files

1. **Download the ViT model weights**:
   ```bash
   # Download from the provided Google Drive link
   # https://drive.google.com/file/d/1XK0dwWCtb17HBnM-bVSAk2_QLmiaMee8/view?usp=drive_link
   # Save as: yolo--object-detection--top-level-view/vit_model.pth
   ```

2. **YOLO model weights**: 
   - The YOLO model weights should be located at: `yolo--object-detection--top-level-view/working/runs/detect/train/weights/best.pt`
   - If you don't have this file, you'll need to train your YOLO model first

### Step 5: Download Test Video

Download the demo video from: https://intip.in/QNpw
Save it as `traffic_test.mp4` inside the `yolo--object-detection--top-level-view` directory.

## 📁 Project Structure

```
car-retrieval-system/
├── yolo--object-detection--top-level-view/
│   ├── compiler_object_detection.py    # Main execution script
│   ├── model/
│   │   └── ViT.py                     # Vision Transformer model
│   ├── working/
│   │   └── runs/detect/train/weights/
│   │       └── best.pt                # YOLO model weights
│   ├── vit_model.pth                  # ViT model weights (download required)
│   └── traffic_test.mp4               # Test video (download required)
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🚀 Usage

### Running the Car Detection and Classification

1. **Ensure all files are in place**:
   - `vit_model.pth` in the `yolo--object-detection--top-level-view` directory
   - `working/runs/detect/train/weights/best.pt` exists inside the yolo directory
   - `traffic_test.mp4` in the `yolo--object-detection--top-level-view` directory

2. **Navigate to the YOLO directory**:
   ```bash
   cd yolo--object-detection--top-level-view
   ```

3. **Run the detection and classification**:
   ```bash
   python compiler_object_detection.py
   ```

4. **Output**:
   - The processed video will be saved as `traffic_test_output_detection_classification.mp4`
   - The script will show progress updates every 100 frames

### Expected Output

The system will:
1. Load the YOLO model for object detection
2. Load the ViT model for car classification
3. Process each frame of the input video
4. Detect car instances using YOLO
5. Classify each detected car using ViT
6. Draw bounding boxes and classification labels on the video
7. Save the annotated video

## ⚙️ Configuration

### Adjusting Detection Confidence

In `compiler_object_detection.py`, you can modify the detection confidence threshold:

```python
# Change confidence threshold (default: 0.5)
results = yolo_model(frame, conf=0.5)  # Adjust this value
```

### Modifying Classification Classes

To add or modify car classes, update the `class_names` list in `compiler_object_detection.py`:

```python
class_names = ['coupe', 'hatchback', 'mpv', 'pickup', 'sedan', 'sports', 'suv', 'truck']
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size or use CPU inference
   - Add `map_location='cpu'` when loading models

2. **Missing Model Files**:
   - Ensure `vit_model.pth` is downloaded and placed in the `yolo--object-detection--top-level-view` directory
   - Verify YOLO weights path: `yolo--object-detection--top-level-view/working/runs/detect/train/weights/best.pt`

3. **Video Processing Issues**:
   - Check if the input video file exists
   - Ensure video codec is supported (MP4 recommended)

4. **Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version compatibility (3.8+)

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed for GPU acceleration
- **Memory Management**: Close other applications to free up system memory
- **Video Resolution**: Consider resizing input video if processing is slow

## 📊 Model Performance

- **Detection Model**: YOLO trained on car detection dataset
- **Classification Model**: ViT trained on Indonesian car dataset
- **Processing Speed**: Depends on hardware and video resolution
- **Accuracy**: Performance metrics available in training logs

## 🔄 Development

### Training Your Own Models

1. **YOLO Training**: Use YOLOv8/YOLOv5 training pipeline
2. **ViT Training**: Modify the ViT model configuration in `model/ViT.py`
3. **Dataset**: Ensure minimum 8 different car types as specified in requirements

### Code Structure

- `compiler_object_detection.py`: Main processing pipeline
- `model/ViT.py`: Vision Transformer implementation
- Modular design allows easy modification of individual components

## 📝 Technical Requirements

- **Minimum 8 car categories** for classification
- **Separate models** for detection and classification (as required)
- **TensorFlow/PyTorch framework** usage (PyTorch implemented)
- **Indonesian car baseline** for classification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the PNU ISLAB Technical AI Test 2025.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the project requirements document
- Ensure all dependencies are properly installed

---

**Note**: This system is designed for the PNU ISLAB Technical AI Test 2025 and demonstrates object detection and classification capabilities on Indonesian car datasets.
