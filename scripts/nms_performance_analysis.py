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
from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if len(boxes) == 0 or len(scores) == 0:
        return []
    
    # Convert to numpy arrays first to handle potential issues
    if not isinstance(boxes, (list, np.ndarray)):
        boxes = [boxes]
    if not isinstance(scores, (list, np.ndarray)):
        scores = [scores]
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    if len(boxes) != len(scores):
        print(f"Warning: boxes ({len(boxes)}) and scores ({len(scores)}) length mismatch")
        return []
    
    if len(boxes) == 0:
        return []
    
    # Convert to tensors
    try:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    except Exception as e:
        print(f"Error converting to tensors: {e}")
        return list(range(len(boxes)))  # Return all indices if conversion fails
    
    try:
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        return keep_indices.cpu().numpy()
    except Exception as e:
        print(f"Error in NMS: {e}")
        return list(range(len(boxes)))  # Return all indices if NMS fails

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

def evaluate_model_with_nms(model_path, val_images_dir, val_labels_dir, conf_threshold=0.25, nms_iou_threshold=0.5):
    """Evaluate model performance with and without NMS."""
    model = YOLO(model_path)
    
    # Get all validation images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(val_images_dir, ext)))
    
    print(f"Evaluating {len(image_files)} validation images...")
    
    # Metrics storage for raw and NMS predictions
    raw_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'total_pred': 0, 'total_gt': 0, 'inference_times': []}
    nms_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'total_pred': 0, 'total_gt': 0, 'nms_removed': 0}
    
    iou_threshold = 0.5
    
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
        
        # Load ground truth
        gt_boxes = load_ground_truth(image_path, val_labels_dir)
        raw_metrics['total_gt'] += len(gt_boxes)
        nms_metrics['total_gt'] += len(gt_boxes)
        
        if len(gt_boxes) == 0:
            continue
        
        # Make prediction
        start_time = time.time()
        results = model(image_path, conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        raw_metrics['inference_times'].append(inference_time)
        
        # Extract raw predictions
        raw_pred_boxes = []
        raw_pred_scores = []
        raw_pred_boxes_xyxy = []  # For NMS
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Convert to normalized YOLO format
                img_h, img_w = result.orig_shape
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to YOLO format for evaluation
                    cx = (x1 + x2) / (2 * img_w)
                    cy = (y1 + y2) / (2 * img_h)
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    
                    raw_pred_boxes.append([cx, cy, w, h])
                    raw_pred_scores.append(conf)
                    raw_pred_boxes_xyxy.append([x1, y1, x2, y2])  # Keep original for NMS
        
        raw_metrics['total_pred'] += len(raw_pred_boxes)
        
        # Apply NMS
        if len(raw_pred_boxes_xyxy) > 0 and len(raw_pred_scores) > 0:
            try:
                keep_indices = apply_nms(raw_pred_boxes_xyxy, raw_pred_scores, nms_iou_threshold)
                nms_pred_boxes = [raw_pred_boxes[i] for i in keep_indices]
                nms_pred_scores = [raw_pred_scores[i] for i in keep_indices]
            except Exception as e:
                print(f"Error applying NMS: {e}")
                # Fallback: use all predictions without NMS
                nms_pred_boxes = raw_pred_boxes
                nms_pred_scores = raw_pred_scores
                keep_indices = list(range(len(raw_pred_boxes)))
        else:
            nms_pred_boxes = []
            nms_pred_scores = []
            keep_indices = []
        
        nms_metrics['total_pred'] += len(nms_pred_boxes)
        nms_metrics['nms_removed'] += len(raw_pred_boxes) - len(nms_pred_boxes)
        
        # Evaluate raw predictions
        raw_tp, raw_fp, raw_fn = evaluate_predictions(raw_pred_boxes, gt_boxes, iou_threshold)
        raw_metrics['tp'] += raw_tp
        raw_metrics['fp'] += raw_fp
        raw_metrics['fn'] += raw_fn
        
        # Evaluate NMS predictions
        nms_tp, nms_fp, nms_fn = evaluate_predictions(nms_pred_boxes, gt_boxes, iou_threshold)
        nms_metrics['tp'] += nms_tp
        nms_metrics['fp'] += nms_fp
        nms_metrics['fn'] += nms_fn
    
    # Calculate final metrics
    def calc_metrics(metrics_dict):
        tp, fp, fn = metrics_dict['tp'], metrics_dict['fp'], metrics_dict['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    raw_precision, raw_recall, raw_f1 = calc_metrics(raw_metrics)
    nms_precision, nms_recall, nms_f1 = calc_metrics(nms_metrics)
    
    results = {
        'model_path': model_path,
        'raw_predictions': {
            'total_pred': raw_metrics['total_pred'],
            'total_gt': raw_metrics['total_gt'],
            'tp': raw_metrics['tp'],
            'fp': raw_metrics['fp'],
            'fn': raw_metrics['fn'],
            'precision': raw_precision,
            'recall': raw_recall,
            'f1_score': raw_f1,
            'avg_inference_time': np.mean(raw_metrics['inference_times']),
            'fps': 1.0 / np.mean(raw_metrics['inference_times'])
        },
        'nms_predictions': {
            'total_pred': nms_metrics['total_pred'],
            'total_gt': nms_metrics['total_gt'],
            'tp': nms_metrics['tp'],
            'fp': nms_metrics['fp'],
            'fn': nms_metrics['fn'],
            'precision': nms_precision,
            'recall': nms_recall,
            'f1_score': nms_f1,
            'boxes_removed': nms_metrics['nms_removed'],
            'reduction_percent': (nms_metrics['nms_removed'] / raw_metrics['total_pred'] * 100) if raw_metrics['total_pred'] > 0 else 0
        },
        'nms_config': {
            'conf_threshold': conf_threshold,
            'nms_iou_threshold': nms_iou_threshold
        }
    }
    
    return results

def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Evaluate predictions against ground truth using IoU matching."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)  # tp, fp, fn
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0  # tp, fp, fn
    
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    return tp, fp, fn

def create_nms_comparison_report(best_results, last_results, output_dir='nms_performance_analysis'):
    """Create comprehensive NMS comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed comparison table
    print("\n" + "=" * 100)
    print(f"{'Model':<12} {'Type':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Total Pred':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'Boxes Removed':<12}")
    print("=" * 100)
    
    for model_name, results in [('Best', best_results), ('Last', last_results)]:
        # Raw predictions
        raw = results['raw_predictions']
        print(f"{model_name:<12} {'Raw':<8} {raw['precision']:<10.3f} {raw['recall']:<8.3f} {raw['f1_score']:<8.3f} {raw['total_pred']:<10} {raw['tp']:<6} {raw['fp']:<6} {raw['fn']:<6} {'-':<12}")
        
        # NMS predictions
        nms = results['nms_predictions']
        print(f"{model_name:<12} {'NMS':<8} {nms['precision']:<10.3f} {nms['recall']:<8.3f} {nms['f1_score']:<8.3f} {nms['total_pred']:<10} {nms['tp']:<6} {nms['fp']:<6} {nms['fn']:<6} {nms['boxes_removed']:<12}")
        print("-" * 100)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Performance metrics comparison
    models = ['Best Model', 'Last Model']
    
    # Raw vs NMS comparison for each model
    for idx, (model_name, results) in enumerate([('Best', best_results), ('Last', last_results)]):
        row = idx
        
        # Precision, Recall, F1 comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        raw_values = [results['raw_predictions']['precision'], 
                     results['raw_predictions']['recall'], 
                     results['raw_predictions']['f1_score']]
        nms_values = [results['nms_predictions']['precision'], 
                     results['nms_predictions']['recall'], 
                     results['nms_predictions']['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[row, 0].bar(x - width/2, raw_values, width, label='Raw', alpha=0.8, color='lightcoral')
        axes[row, 0].bar(x + width/2, nms_values, width, label='NMS', alpha=0.8, color='skyblue')
        axes[row, 0].set_xlabel('Metrics')
        axes[row, 0].set_ylabel('Score')
        axes[row, 0].set_title(f'{model_name} Model: Raw vs NMS Performance')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(metrics)
        axes[row, 0].legend()
        axes[row, 0].set_ylim(0, 1)
        
        # Detection counts
        detection_types = ['TP', 'FP', 'FN']
        raw_counts = [results['raw_predictions']['tp'], 
                     results['raw_predictions']['fp'], 
                     results['raw_predictions']['fn']]
        nms_counts = [results['nms_predictions']['tp'], 
                     results['nms_predictions']['fp'], 
                     results['nms_predictions']['fn']]
        
        x = np.arange(len(detection_types))
        axes[row, 1].bar(x - width/2, raw_counts, width, label='Raw', alpha=0.8, color='lightcoral')
        axes[row, 1].bar(x + width/2, nms_counts, width, label='NMS', alpha=0.8, color='skyblue')
        axes[row, 1].set_xlabel('Detection Type')
        axes[row, 1].set_ylabel('Count')
        axes[row, 1].set_title(f'{model_name} Model: Detection Counts')
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(detection_types)
        axes[row, 1].legend()
        
        # Box reduction visualization
        total_boxes = results['raw_predictions']['total_pred']
        remaining_boxes = results['nms_predictions']['total_pred']
        removed_boxes = results['nms_predictions']['boxes_removed']
        
        axes[row, 2].pie([remaining_boxes, removed_boxes], 
                        labels=[f'Kept ({remaining_boxes})', f'Removed ({removed_boxes})'],
                        autopct='%1.1f%%', startangle=90,
                        colors=['lightgreen', 'lightcoral'])
        axes[row, 2].set_title(f'{model_name} Model: NMS Box Reduction\n({results["nms_predictions"]["reduction_percent"]:.1f}% removed)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nms_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    detailed_results = {
        'best_model': best_results,
        'last_model': last_results,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'nms_detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, 'nms_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("Model,Type,Precision,Recall,F1_Score,Total_Predictions,TP,FP,FN,Boxes_Removed,Reduction_Percent\n")
        
        for model_name, results in [('Best', best_results), ('Last', last_results)]:
            # Raw
            raw = results['raw_predictions']
            f.write(f"{model_name},Raw,{raw['precision']:.4f},{raw['recall']:.4f},{raw['f1_score']:.4f},"
                   f"{raw['total_pred']},{raw['tp']},{raw['fp']},{raw['fn']},0,0.0\n")
            
            # NMS
            nms = results['nms_predictions']
            f.write(f"{model_name},NMS,{nms['precision']:.4f},{nms['recall']:.4f},{nms['f1_score']:.4f},"
                   f"{nms['total_pred']},{nms['tp']},{nms['fp']},{nms['fn']},{nms['boxes_removed']},{nms['reduction_percent']:.2f}\n")
    
    return detailed_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Performance Analysis with NMS Evaluation')
    parser.add_argument('--val_images', type=str, default='dataset/images/val',
                       help='Validation images directory')
    parser.add_argument('--val_labels', type=str, default='dataset/labels/val',
                       help='Validation labels directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--nms_iou', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='nms_performance_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔍 YOLO Model Performance Analysis with NMS")
    print("=" * 60)
    
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
    print(f"🔧 NMS IoU threshold: {args.nms_iou}")
    print()
    
    # Evaluate models with NMS
    print("🚀 Evaluating Best Model (Raw + NMS)...")
    best_results = evaluate_model_with_nms(best_model_path, args.val_images, args.val_labels, args.conf, args.nms_iou)
    
    print("🚀 Evaluating Last Model (Raw + NMS)...")
    last_results = evaluate_model_with_nms(last_model_path, args.val_images, args.val_labels, args.conf, args.nms_iou)
    
    # Create comparison report
    print("📈 Creating NMS comparison report...")
    detailed_results = create_nms_comparison_report(best_results, last_results, args.output_dir)
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}/")
    print("✅ NMS analysis complete!")

if __name__ == '__main__':
    main()