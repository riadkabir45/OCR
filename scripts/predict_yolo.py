from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import glob

def predict_and_visualize(model_path, image_path, conf_threshold=0.25):
    """
    Run YOLO prediction on an image and visualize results.
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
    """
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(image_path, conf=conf_threshold)
    
    # Load image for visualization
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Draw rectangle
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                edgecolor='red', facecolor='none', linewidth=2))
                
                # Add confidence score
                plt.text(x1, y1-10, f'{conf:.2f}', 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                        fontsize=8, color='white')
    
    plt.title(f'YOLO Predictions - {os.path.basename(image_path)}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return results

def find_latest_model():
    """Find the latest trained YOLO model."""
    # Look for models in runs/detect/train*/weights/
    model_paths = glob.glob('runs/detect/train*/weights/best.pt')
    if not model_paths:
        model_paths = glob.glob('runs/detect/train*/weights/last.pt')
    
    if model_paths:
        # Sort by modification time, return latest
        latest_model = max(model_paths, key=os.path.getmtime)
        return latest_model
    else:
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run YOLO prediction and visualization')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to YOLO model (.pt file). If None, uses latest trained model.')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Find model
    if args.model is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No trained model found. Please specify --model path or train a model first.")
            exit(1)
        print(f"Using latest model: {model_path}")
    else:
        model_path = args.model
    
    # Run prediction
    results = predict_and_visualize(model_path, args.image, args.conf)
    
    print(f"Prediction completed. Found {len(results[0].boxes) if results[0].boxes else 0} boxes.")