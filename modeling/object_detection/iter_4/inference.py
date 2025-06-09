import cv2
import numpy as np
from ultralytics import YOLO
import torch


class CarDetectionSystem:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Initialize the car detection system

        Args:
            model_path (str): Path to your trained YOLO model (.pt file)
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def load_model_from_config(self, config_path, weights_path):
        """
        Alternative method to load model from config and weights

        Args:
            config_path (str): Path to yolo.yaml
            weights_path (str): Path to .pt weights file
        """
        # Load model with custom config
        self.model = YOLO(config_path)
        self.model.load(weights_path)

    def detect_cars(self, image_path_or_array, save_results=False):
        """
        Detect cars in an image

        Args:
            image_path_or_array: Path to image file or numpy array
            save_results (bool): Whether to save annotated results

        Returns:
            dict: Detection results with bounding boxes, confidence scores, etc.
        """
        # Run inference
        results = self.model(
            image_path_or_array,
            conf=self.conf_threshold,
            save=save_results,
            classes=[2, 5, 7]  # COCO classes for car, bus, truck (adjust as needed)
        )

        return results

    def extract_car_crops(self, image, results):
        """
        Extract cropped car images from detection results

        Args:
            image (np.array): Original image
            results: YOLO detection results

        Returns:
            list: List of cropped car images
        """
        car_crops = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Extract crop
                    car_crop = image[y1:y2, x1:x2]
                    car_crops.append({
                        'image': car_crop,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': box.conf[0].cpu().numpy()
                    })

        return car_crops


# Example usage for your project

def main():
    # Initialize detection system with your trained model
    # Option 1: Load directly from .pt file
    detector = CarDetectionSystem('yolo11n.pt')  # or 'yolo8m.pt'

    # Option 2: Load from config and weights (if you have custom config)
    # detector = CarDetectionSystem('yolo11n.pt')
    # detector.load_model_from_config('yolo.yaml', 'yolo11n.pt')

    # For video inference (as required in your project)
    video_path = "path_to_your_video.mp4"  # Replace with your video URL
    cap = cv2.VideoCapture(video_path)

    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on current frame
        results = detector.detect_cars(frame)

        # Extract car crops for classification
        car_crops = detector.extract_car_crops(frame, results)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Display frame count and detections
        cv2.putText(annotated_frame, f'Frame: {frame_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Cars detected: {len(car_crops)}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(annotated_frame)

        # Optional: Display frame (comment out for batch processing)
        # cv2.imshow('Car Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_count += 1

        # Process car crops with your classification model here
        for i, crop_data in enumerate(car_crops):
            crop_image = crop_data['image']
            # TODO: Pass crop_image to your car classification model
            # car_type = classify_car(crop_image)
            print(f"Frame {frame_count}, Car {i + 1}: bbox={crop_data['bbox']}, conf={crop_data['confidence']:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# For single image inference
def single_image_inference():
    detector = CarDetectionSystem('yolo11n.pt')

    # Load image
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)

    # Run detection
    results = detector.detect_cars(image_path, save_results=True)

    # Extract car crops
    car_crops = detector.extract_car_crops(image, results)

    # Print results
    print(f"Detected {len(car_crops)} cars in the image")
    for i, crop_data in enumerate(car_crops):
        print(f"Car {i + 1}: bbox={crop_data['bbox']}, confidence={crop_data['confidence']:.2f}")

    return car_crops


if __name__ == "__main__":
    #main()
    single_image_inference()