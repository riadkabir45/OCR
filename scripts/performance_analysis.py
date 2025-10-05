import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import os
import json
from pathlib import Path
import time
from sklearn.metrics import precision_recall_curve, average_precision_score

def find_models():
    """Find the best and last trained YOLO models."""
    best_paths = glob.glob('runs/detect/train*/weights/best.pt')
    last_paths = glob.glob('runs/detect/train*/weights/last.pt')
    
    best_model = max(best_paths, key=os.path.getmtime) if best_paths else None
    last_model = max(last_paths, key=os.path.getmtime) if last_paths else None
    
    return best_model, last_model

def load_ground_truth(image_path, labels_dir):
    """Load ground truth bounding boxes for an image."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(labels_dir, base_name + '.txt')
    
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class_id center_x center_y width height (normalized)
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx, cy, w, h])
    
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in YOLO format (cx, cy, w, h)."""
    # Convert to corner coordinates
    def yolo_to_corners(cx, cy, w, h):
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(*box1)
    x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(*box2)
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_model_performance(model_path, val_images_dir, val_labels_dir, conf_threshold=0.25):
    """Evaluate model performance on validation set."""
    model = YOLO(model_path)
    
    # Get all validation images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(val_images_dir, ext)))
    
    print(f"Evaluating {len(image_files)} validation images...")
    
    # Metrics storage
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_aps = []
    inference_times = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    iou_threshold = 0.5
    
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
        
        # Load ground truth
        gt_boxes = load_ground_truth(image_path, val_labels_dir)
        total_gt_boxes += len(gt_boxes)
        
        if len(gt_boxes) == 0:
            continue
        
        # Make prediction
        start_time = time.time()
        results = model(image_path, conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Extract predictions
        pred_boxes = []
        pred_scores = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Convert to normalized YOLO format
                img_h, img_w = result.orig_shape
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to YOLO format
                    cx = (x1 + x2) / (2 * img_w)
                    cy = (y1 + y2) / (2 * img_h)
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    
                    pred_boxes.append([cx, cy, w, h])
                    pred_scores.append(conf)
        
        total_pred_boxes += len(pred_boxes)
        
        # Calculate matches using IoU
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)
        
        tp = 0
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[i] = True
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate metrics for this image
        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        
        # Calculate AP for this image
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Create binary labels (1 for TP, 0 for FP)
            y_true = [1 if matched else 0 for matched in pred_matched]
            ap = average_precision_score(y_true, pred_scores) if len(set(y_true)) > 1 else 0
            all_aps.append(ap)
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics = {
        'model_path': model_path,
        'total_images': len(image_files),
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'mean_precision': np.mean(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'mean_f1': np.mean(all_f1_scores),
        'mean_ap': np.mean(all_aps) if all_aps else 0,
        'avg_inference_time': np.mean(inference_times),
        'fps': 1.0 / np.mean(inference_times),
        'precision_std': np.std(all_precisions),
        'recall_std': np.std(all_recalls),
        'f1_std': np.std(all_f1_scores)
    }
    
    return metrics

def create_comparison_report(best_metrics, last_metrics, output_dir='performance_analysis'):
    """Create comprehensive comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = {
        'Metric': [
            'Overall Precision', 'Overall Recall', 'Overall F1-Score',
            'Mean Precision', 'Mean Recall', 'Mean F1-Score', 'Mean AP',
            'Total GT Boxes', 'Total Pred Boxes', 'True Positives',
            'False Positives', 'False Negatives', 'Avg Inference Time (s)',
            'FPS', 'Precision Std', 'Recall Std', 'F1 Std'
        ],
        'Best Model': [
            f"{best_metrics['overall_precision']:.4f}",
            f"{best_metrics['overall_recall']:.4f}",
            f"{best_metrics['overall_f1']:.4f}",
            f"{best_metrics['mean_precision']:.4f}",
            f"{best_metrics['mean_recall']:.4f}",
            f"{best_metrics['mean_f1']:.4f}",
            f"{best_metrics['mean_ap']:.4f}",
            best_metrics['total_gt_boxes'],
            best_metrics['total_pred_boxes'],
            best_metrics['total_tp'],
            best_metrics['total_fp'],
            best_metrics['total_fn'],
            f"{best_metrics['avg_inference_time']:.4f}",
            f"{best_metrics['fps']:.2f}",
            f"{best_metrics['precision_std']:.4f}",
            f"{best_metrics['recall_std']:.4f}",
            f"{best_metrics['f1_std']:.4f}"
        ],
        'Last Model': [
            f"{last_metrics['overall_precision']:.4f}",
            f"{last_metrics['overall_recall']:.4f}",
            f"{last_metrics['overall_f1']:.4f}",
            f"{last_metrics['mean_precision']:.4f}",
            f"{last_metrics['mean_recall']:.4f}",
            f"{last_metrics['mean_f1']:.4f}",
            f"{last_metrics['mean_ap']:.4f}",
            last_metrics['total_gt_boxes'],
            last_metrics['total_pred_boxes'],
            last_metrics['total_tp'],
            last_metrics['total_fp'],
            last_metrics['total_fn'],
            f"{last_metrics['avg_inference_time']:.4f}",
            f"{last_metrics['fps']:.2f}",
            f"{last_metrics['precision_std']:.4f}",
            f"{last_metrics['recall_std']:.4f}",
            f"{last_metrics['f1_std']:.4f}"
        ]
    }
    
    # Create simple comparison table without pandas
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} {'Best Model':<15} {'Last Model':<15}")
    print("=" * 80)
    
    for i, metric in enumerate(comparison_data['Metric']):
        print(f"{metric:<25} {comparison_data['Best Model'][i]:<15} {comparison_data['Last Model'][i]:<15}")
    
    # Save to CSV manually
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("Metric,Best Model,Last Model\n")
        for i, metric in enumerate(comparison_data['Metric']):
            f.write(f"{metric},{comparison_data['Best Model'][i]},{comparison_data['Last Model'][i]}\n")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance metrics comparison
    metrics_to_plot = ['overall_precision', 'overall_recall', 'overall_f1', 'mean_ap']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Mean AP']
    
    best_values = [best_metrics[m] for m in metrics_to_plot]
    last_values = [last_metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, best_values, width, label='Best Model', alpha=0.8)
    axes[0, 0].bar(x + width/2, last_values, width, label='Last Model', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_names)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # Speed comparison
    speed_metrics = ['avg_inference_time', 'fps']
    speed_names = ['Avg Inference Time (s)', 'FPS']
    
    # Normalize for comparison (inference time is inverse of speed)
    best_speed = [best_metrics['avg_inference_time'], best_metrics['fps']]
    last_speed = [last_metrics['avg_inference_time'], last_metrics['fps']]
    
    axes[0, 1].bar(['Best Model', 'Last Model'], [best_metrics['fps'], last_metrics['fps']], 
                   color=['blue', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('FPS')
    axes[0, 1].set_title('Inference Speed Comparison')
    
    # Detection counts
    detection_metrics = ['total_tp', 'total_fp', 'total_fn']
    detection_names = ['True Positives', 'False Positives', 'False Negatives']
    
    best_detections = [best_metrics[m] for m in detection_metrics]
    last_detections = [last_metrics[m] for m in detection_metrics]
    
    x = np.arange(len(detection_names))
    axes[1, 0].bar(x - width/2, best_detections, width, label='Best Model', alpha=0.8)
    axes[1, 0].bar(x + width/2, last_detections, width, label='Last Model', alpha=0.8)
    axes[1, 0].set_xlabel('Detection Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Detection Counts Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(detection_names)
    axes[1, 0].legend()
    
    # Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    summary_data = [
        ['Model', 'Precision', 'Recall', 'F1', 'FPS'],
        ['Best', f"{best_metrics['overall_precision']:.3f}", 
         f"{best_metrics['overall_recall']:.3f}", 
         f"{best_metrics['overall_f1']:.3f}",
         f"{best_metrics['fps']:.1f}"],
        ['Last', f"{last_metrics['overall_precision']:.3f}", 
         f"{last_metrics['overall_recall']:.3f}", 
         f"{last_metrics['overall_f1']:.3f}",
         f"{last_metrics['fps']:.1f}"]
    ]
    
    table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0],
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed metrics to JSON
    detailed_metrics = {
        'best_model': best_metrics,
        'last_model': last_metrics,
        'comparison_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    return comparison_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive performance analysis of YOLO models')
    parser.add_argument('--val_images', type=str, default='dataset/images/val',
                       help='Validation images directory')
    parser.add_argument('--val_labels', type=str, default='dataset/labels/val',
                       help='Validation labels directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--output_dir', type=str, default='performance_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔍 YOLO Model Performance Analysis")
    print("=" * 50)
    
    # Find models
    best_model_path, last_model_path = find_models()
    
    if not best_model_path or not last_model_path:
        print("❌ Could not find both best and last models!")
        return
    
    print(f"📊 Best model: {best_model_path}")
    print(f"📊 Last model: {last_model_path}")
    print(f"📁 Validation images: {args.val_images}")
    print(f"📁 Validation labels: {args.val_labels}")
    print(f"🎯 Confidence threshold: {args.conf}")
    print()
    
    # Evaluate models
    print("🚀 Evaluating Best Model...")
    best_metrics = evaluate_model_performance(best_model_path, args.val_images, args.val_labels, args.conf)
    
    print("🚀 Evaluating Last Model...")
    last_metrics = evaluate_model_performance(last_model_path, args.val_images, args.val_labels, args.conf)
    
    # Create comparison report
    print("📈 Creating comparison report...")
    comparison_data = create_comparison_report(best_metrics, last_metrics, args.output_dir)
    
    print("\n" + "=" * 50)
    print("📋 PERFORMANCE SUMMARY")
    print("=" * 50)
    # Summary already printed in create_comparison_report function
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}/")
    print("✅ Analysis complete!")

if __name__ == '__main__':
    main()