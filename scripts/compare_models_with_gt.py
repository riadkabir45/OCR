from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import glob
import random
import torch
import numpy as np
from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    
    keep_indices = nms(boxes, scores, iou_threshold)
    return keep_indices.cpu().numpy()

def find_models():
    """Find the best and last trained YOLO models."""
    best_paths = glob.glob('runs/detect/train*/weights/best.pt')
    last_paths = glob.glob('runs/detect/train*/weights/last.pt')
    
    best_model = max(best_paths, key=os.path.getmtime) if best_paths else None
    last_model = max(last_paths, key=os.path.getmtime) if last_paths else None
    
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

def predict_with_nms(model, image_path, conf_threshold=0.25, iou_threshold=0.5):
    """Run prediction with NMS to remove overlapping boxes."""
    results = model(image_path, conf=conf_threshold)
    
    all_boxes = []
    all_scores = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            
            all_boxes.extend(xyxy)
            all_scores.extend(conf)
    
    if len(all_boxes) == 0:
        return [], []
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    
    keep_indices = apply_nms(all_boxes, all_scores, iou_threshold)
    
    filtered_boxes = all_boxes[keep_indices]
    filtered_scores = all_scores[keep_indices]
    
    return filtered_boxes, filtered_scores

def predict_and_compare_with_ground_truth(best_model_path, last_model_path, image_path, labels_dir,
                                        conf_threshold=0.25, iou_threshold=0.5):
    """Run predictions with both models, apply NMS, and compare with ground truth."""
    # Load models
    best_model = YOLO(best_model_path) if best_model_path else None
    last_model = YOLO(last_model_path) if last_model_path else None
    
    # Load image and ground truth
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_boxes = load_ground_truth(image_path, labels_dir)
    
    # Create subplots: 3 rows x 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    # Row 1: Ground Truth
    axes[0, 0].imshow(image_rgb)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          edgecolor='green', facecolor='none', linewidth=2))
    axes[0, 0].set_title(f'Ground Truth - {len(gt_boxes)} boxes')
    axes[0, 0].axis('off')
    
    # Empty cells for symmetry
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    # Row 2: Best Model
    if best_model:
        # Raw predictions
        best_results_raw = best_model(image_path, conf=conf_threshold)
        axes[1, 0].imshow(image_rgb)
        
        best_boxes_raw = 0
        for result in best_results_raw:
            boxes = result.boxes
            if boxes is not None:
                best_boxes_raw = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    axes[1, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                      edgecolor='red', facecolor='none', linewidth=1))
                    axes[1, 0].text(x1, y1-10, f'{conf:.2f}', 
                                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                                   fontsize=6, color='white')
        
        axes[1, 0].set_title(f'Best Model (Raw) - {best_boxes_raw} boxes')
        axes[1, 0].axis('off')
        
        # NMS filtered predictions
        best_boxes_nms, best_scores_nms = predict_with_nms(best_model, image_path, conf_threshold, iou_threshold)
        axes[1, 1].imshow(image_rgb)
        
        for box, score in zip(best_boxes_nms, best_scores_nms):
            x1, y1, x2, y2 = box
            axes[1, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='red', facecolor='none', linewidth=2))
            axes[1, 1].text(x1, y1-10, f'{score:.2f}', 
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                           fontsize=8, color='white')
        
        axes[1, 1].set_title(f'Best Model (NMS) - {len(best_boxes_nms)} boxes')
        axes[1, 1].axis('off')
        
        # Overlay with ground truth
        axes[1, 2].imshow(image_rgb)
        # Ground truth in green
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            axes[1, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='green', facecolor='none', linewidth=2, alpha=0.7))
        # Predictions in red
        for box, score in zip(best_boxes_nms, best_scores_nms):
            x1, y1, x2, y2 = box
            axes[1, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='red', facecolor='none', linewidth=2, linestyle='--'))
        
        axes[1, 2].set_title(f'Best vs GT (Green=GT, Red=Pred)')
        axes[1, 2].axis('off')
    else:
        for col in range(3):
            axes[1, col].text(0.5, 0.5, 'Best model not found', ha='center', va='center', transform=axes[1, col].transAxes)
            axes[1, col].set_title('Best Model - Not Available')
            axes[1, col].axis('off')
    
    # Row 3: Last Model
    if last_model:
        # Raw predictions
        last_results_raw = last_model(image_path, conf=conf_threshold)
        axes[2, 0].imshow(image_rgb)
        
        last_boxes_raw = 0
        for result in last_results_raw:
            boxes = result.boxes
            if boxes is not None:
                last_boxes_raw = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    axes[2, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                      edgecolor='blue', facecolor='none', linewidth=1))
                    axes[2, 0].text(x1, y1-10, f'{conf:.2f}', 
                                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                                   fontsize=6, color='white')
        
        axes[2, 0].set_title(f'Last Model (Raw) - {last_boxes_raw} boxes')
        axes[2, 0].axis('off')
        
        # NMS filtered predictions
        last_boxes_nms, last_scores_nms = predict_with_nms(last_model, image_path, conf_threshold, iou_threshold)
        axes[2, 1].imshow(image_rgb)
        
        for box, score in zip(last_boxes_nms, last_scores_nms):
            x1, y1, x2, y2 = box
            axes[2, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='blue', facecolor='none', linewidth=2))
            axes[2, 1].text(x1, y1-10, f'{score:.2f}', 
                           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                           fontsize=8, color='white')
        
        axes[2, 1].set_title(f'Last Model (NMS) - {len(last_boxes_nms)} boxes')
        axes[2, 1].axis('off')
        
        # Overlay with ground truth
        axes[2, 2].imshow(image_rgb)
        # Ground truth in green
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            axes[2, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='green', facecolor='none', linewidth=2, alpha=0.7))
        # Predictions in blue
        for box, score in zip(last_boxes_nms, last_scores_nms):
            x1, y1, x2, y2 = box
            axes[2, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='blue', facecolor='none', linewidth=2, linestyle='--'))
        
        axes[2, 2].set_title(f'Last vs GT (Green=GT, Blue=Pred)')
        axes[2, 2].axis('off')
    else:
        for col in range(3):
            axes[2, col].text(0.5, 0.5, 'Last model not found', ha='center', va='center', transform=axes[2, col].transAxes)
            axes[2, col].set_title('Last Model - Not Available')
            axes[2, col].axis('off')
    
    plt.suptitle(f'Model Comparison with Ground Truth (IoU={iou_threshold})\n{os.path.basename(image_path)}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        'gt_boxes': len(gt_boxes),
        'best_raw': best_boxes_raw if best_model else 0,
        'best_nms': len(best_boxes_nms) if best_model else 0,
        'last_raw': last_boxes_raw if last_model else 0,
        'last_nms': len(last_boxes_nms) if last_model else 0
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare YOLO models with ground truth and NMS')
    parser.add_argument('--image_dir', type=str, default='dataset/images/val', 
                       help='Directory containing validation images')
    parser.add_argument('--labels_dir', type=str, default='dataset/labels/val', 
                       help='Directory containing validation labels')
    parser.add_argument('--image_name', type=str, default=None, 
                       help='Specific image name to analyze (if None, picks random)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5, 
                       help='IoU threshold for NMS (higher = more aggressive merging)')
    
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
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold for NMS: {args.iou}")
    if best_model:
        print(f"Best model: {best_model}")
    if last_model:
        print(f"Last model: {last_model}")
    
    # Run comparison with ground truth
    results = predict_and_compare_with_ground_truth(
        best_model, last_model, selected_image, args.labels_dir, args.conf, args.iou)
    
    print(f"\nResults:")
    print(f"  Ground Truth: {results['gt_boxes']} boxes")
    print(f"  Best model: {results['best_raw']} → {results['best_nms']} boxes (after NMS)")
    print(f"  Last model: {results['last_raw']} → {results['last_nms']} boxes (after NMS)")
    if results['best_raw'] > 0:
        print(f"  Best reduction: {results['best_raw'] - results['best_nms']} overlapping boxes removed")
    if results['last_raw'] > 0:
        print(f"  Last reduction: {results['last_raw'] - results['last_nms']} overlapping boxes removed")