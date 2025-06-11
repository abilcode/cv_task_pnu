import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model.ViT import ViT

# Load YOLO model
yolo_model = YOLO("working/runs/detect/train/weights/best.pt")

# Load ViT model
checkpoint = torch.load('vit_model.pth', map_location='cpu')
vit_model = ViT(**checkpoint['model_config'])
vit_model.load_state_dict(checkpoint['model_state_dict'])
vit_model.eval()

# ViT preprocessing
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (update with your actual class names)
class_names = ['coupe', 'hatchback', 'mpv', 'pickup', 'sedan', 'sports', 'suv', 'wagon']


def classify_crop(crop_image):
    """Classify cropped image using ViT"""
    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)

    # Transform and predict
    input_tensor = vit_transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = vit_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


# Video processing
input_video = 'traffic_test.mp4'
output_video = 'traffic_test_output_detection_classification.mp4'

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
print("Processing video with detection + classification...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame, conf=0.5)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Crop detected object
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                # Classify the crop
                predicted_class, confidence = classify_crop(crop)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add classification label
                label = f"{predicted_class}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add frame counter
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write frame
    out.write(frame)
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Output saved as: {output_video}")
print(f"Total frames processed: {frame_count}")