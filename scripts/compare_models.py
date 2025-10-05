from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import glob
import random

def find_models():
    """Find the best and last trained YOLO models."""
    # Look for models in runs/detect/train*/weights/
    best_paths = glob.glob('runs/detect/train*/weights/best.pt')
    last_paths = glob.glob('runs/detect/train*/weights/last.pt')
    
    if best_paths:
        best_model = max(best_paths, key=os.path.getmtime)
    else:
        best_model = None
        
    if last_paths:
        last_model = max(last_paths, key=os.path.getmtime)
    else:
        last_model = None
    
    return best_model, last_model

def get_image(image_dir, image_name=None):
    """Get a specific image or random image from the directory."""
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.png')))
    
    if not image_files:
        return None
    
    if image_name:
        # Look for specific image
        for img_path in image_files:
            if image_name.lower() in os.path.basename(img_path).lower():
                return img_path
        print(f"Warning: Image '{image_name}' not found, selecting random image")
    
    return random.choice(image_files)

def load_ground_truth(image_path, labels_dir):
    """Load ground truth bounding boxes for an image."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(labels_dir, base_name + '.txt')
    
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class_id center_x center_y width height (normalized)
                _, cx, cy, w, h = map(float, parts[:5])
                
                # Convert to pixel coordinates
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                x2 = (cx + w/2) * img_w
                y2 = (cy + h/2) * img_h
                
                boxes.append([x1, y1, x2, y2])
    
    return boxes

def predict_and_compare(best_model_path, last_model_path, image_path, labels_dir, conf_threshold=0.25):
    """
    Run predictions with both best and last models and compare with ground truth.
    """
    # Load models
    best_model = YOLO(best_model_path) if best_model_path else None
    last_model = YOLO(last_model_path) if last_model_path else None
    
    # Load image and ground truth
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_boxes = load_ground_truth(image_path, labels_dir)
    
    # Create subplots: 1 row x 3 columns (GT, Best, Last)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Ground Truth
    axes[0].imshow(image_rgb)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       edgecolor='green', facecolor='none', linewidth=2))
    axes[0].set_title(f'Ground Truth - {len(gt_boxes)} boxes')
    axes[0].axis('off')
    
    # Predict with best model
    if best_model:
        best_results = best_model(image_path, conf=conf_threshold)
        axes[1].imshow(image_rgb)
        
        best_boxes = 0
        for result in best_results:
            boxes = result.boxes
            if boxes is not None:
                best_boxes = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    axes[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   edgecolor='red', facecolor='none', linewidth=2))
                    axes[1].text(x1, y1-10, f'{conf:.2f}', 
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                               fontsize=8, color='white')
        
        axes[1].set_title(f'Best Model - {best_boxes} boxes detected\n{os.path.basename(best_model_path)}')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Best model not found', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Best Model - Not Available')
        axes[1].axis('off')
        best_boxes = 0
    
    # Predict with last model
    if last_model:
        last_results = last_model(image_path, conf=conf_threshold)
        axes[2].imshow(image_rgb)
        
        last_boxes = 0
        for result in last_results:
            boxes = result.boxes
            if boxes is not None:
                last_boxes = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    axes[2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   edgecolor='blue', facecolor='none', linewidth=2))
                    axes[2].text(x1, y1-10, f'{conf:.2f}', 
                               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                               fontsize=8, color='white')
        
        axes[2].set_title(f'Last Model - {last_boxes} boxes detected\n{os.path.basename(last_model_path)}')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'Last model not found', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Last Model - Not Available')
        axes[2].axis('off')
        last_boxes = 0
    
    plt.suptitle(f'Model Comparison with Ground Truth\n{os.path.basename(image_path)}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return len(gt_boxes), best_boxes, last_boxes

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare YOLO models with ground truth')
    parser.add_argument('--image_dir', type=str, default='dataset/images/val', 
                       help='Directory containing validation images')
    parser.add_argument('--labels_dir', type=str, default='dataset/labels/val', 
                       help='Directory containing validation labels')
    parser.add_argument('--image_name', type=str, default=None, 
                       help='Specific image name to analyze (if None, picks random)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Find models
    best_model, last_model = find_models()
    
    if not best_model and not last_model:
        print("No trained models found. Please train a model first.")
        exit(1)
    
    # Get image
    selected_image = get_image(args.image_dir, args.image_name)
    
    if not selected_image:
        print(f"No images found in {args.image_dir}")
        exit(1)
    
    print(f"Selected image: {selected_image}")
    if best_model:
        print(f"Best model: {best_model}")
    if last_model:
        print(f"Last model: {last_model}")
    
    # Run comparison
    gt_count, best_count, last_count = predict_and_compare(best_model, last_model, selected_image, args.labels_dir, args.conf)
    
    print(f"Results:")
    print(f"  Ground Truth: {gt_count} boxes")
    print(f"  Best model: {best_count} boxes detected")
    print(f"  Last model: {last_count} boxes detected")