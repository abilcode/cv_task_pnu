# Car Detection System with ResNet50 Backbone

A simple and readable car object detection system using PyTorch and Faster R-CNN with ResNet50 backbone. This project fulfills the requirements for the PNU ISLAB technical AI test 2025.

## 🚗 Project Overview

This project implements a car detection system that can:
- Detect multiple car instances in images
- Process single images, batch images, and videos
- Use ResNet50 as the backbone network through Faster R-CNN
- Provide clean, readable, and well-documented code

## 📁 Project Structure

```
cv_task_pnu/
├── modeling/
│   ├── data/
│   │   ├── training_images/          # Training images
│   │   ├── testing_images/           # Testing images
│   │   ├── train_solution_bounding_boxes.csv  # Annotations
│   │   └── sample_submission.csv
│   └── object_detection/
├── car_detection_training.py         # Training script
├── car_detection_inference.py        # Inference script
├── data_preparation.py               # Data analysis and preparation
├── requirements.txt                  # Dependencies
├── main.py                          # Main entry point
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For M-series Mac users, also install:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Data Preparation

First, check your data structure and prepare the dataset:

```python
python data_preparation.py
```

This script will:
- ✅ Check if your data structure is correct
- 📊 Analyze your annotations
- 🔍 Validate images and bounding boxes
- 📈 Generate visualizations
- 📂 Create train/validation splits

### 3. Training the Model

```python
python car_detection_training.py
```

Training features:
- 🏗️ **Architecture**: Faster R-CNN with ResNet50 backbone
- 🎯 **Task**: Car object detection (binary: background vs car)
- 📊 **Monitoring**: Real-time loss tracking and visualization
- 💾 **Checkpoints**: Automatic model saving every 2 epochs
- 📈 **Visualization**: Training loss plots and sample predictions

### 4. Running Inference

```python
python car_detection_inference.py
```

Inference capabilities:
- 🖼️ **Single Image**: Detect cars in one image
- 📁 **Batch Processing**: Process multiple images
- 🎬 **Video Processing**: Real-time car detection in videos
- 📊 **Results**: Automatic visualization and saving

## 📋 Data Format Requirements

Your annotations CSV should have these columns:
- `image_path`: Relative path to image
- `xmin`, `ymin`, `xmax`, `ymax`: Bounding box coordinates
- `class_name` (optional): For compatibility

Example:
```csv
image_path,xmin,ymin,xmax,ymax,class_name
car001.jpg,100,50,300,200,car
car001.jpg,350,100,500,250,car
car002.jpg,80,60,280,180,car
```

## 🛠️ Model Architecture

```
Faster R-CNN with ResNet50 Backbone
├── Backbone: ResNet50 (pre-trained on ImageNet)
├── Feature Pyramid Network (FPN)
├── Region Proposal Network (RPN)
└── ROI Head with FastRCNN Predictor
    ├── Input Features: 1024
    └── Output Classes: 2 (background + car)
```

## ⚙️ Configuration Options

### Training Configuration
```python
config = {
    'batch_size': 4,           # Adjust based on GPU memory
    'num_epochs': 10,          # Training epochs
    'learning_rate': 0.005,    # Learning rate
    'momentum': 0.9,           # SGD momentum
    'weight_decay': 0.0005,    # L2 regularization
    'num_classes': 2,          # background + car
}
```

### Inference Configuration
```python
confidence_threshold = 0.5     # Detection confidence threshold
device = 'cuda' or 'cpu'       # Automatic device selection
```

## 📊 Performance Monitoring

The training script automatically generates:

1. **Training Loss Plot** (`training_loss.png`)
   - Real-time loss tracking
   - Saved automatically after training

2. **Detection Visualizations** (`detection_results.png`)
   - Sample predictions with ground truth
   - Green boxes: Ground truth
   - Red boxes: Predictions

3. **Model Checkpoints**
   - `checkpoint_epoch_X.pth`: Intermediate checkpoints
   -