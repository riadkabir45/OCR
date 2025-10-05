import os
import json
import glob

def convert_json_to_yolo(json_dir, yolo_dir, class_map=None):
    """Convert JSON annotations to YOLO format.
    
    Args:
        json_dir: Directory containing JSON annotation files
        yolo_dir: Output directory for YOLO txt labels
        class_map: Optional mapping of class names to IDs
    """
    os.makedirs(yolo_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get corresponding txt file name
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        yolo_txt = os.path.join(yolo_dir, base_name + '.txt')
        
        # Get image dimensions
        img_w = data.get('imageWidth', 1)
        img_h = data.get('imageHeight', 1)
        
        # Extract all bounding boxes
        shapes = data.get('shapes', [])
        lines = []
        
        for shape in shapes:
            points = shape.get('points', [])
            if len(points) == 2:
                # Extract coordinates
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Convert to YOLO format: center_x, center_y, width, height (normalized)
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = abs(x2 - x1) / img_w
                h = abs(y2 - y1) / img_h
                
                # For handwritten text, we use class 0 (single class detection)
                class_id = 0
                
                # Format: class_id center_x center_y width height
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # Write YOLO format file
        with open(yolo_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    print(f"Converted {len(json_files)} JSON files to YOLO format in {yolo_dir}")
    if json_files:
        print(f"Example: {json_files[0]} -> {len(lines)} boxes")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert JSON annotations to YOLO format')
    parser.add_argument('--json_dir', type=str, default='converted', 
                       help='Directory with JSON annotations')
    parser.add_argument('--yolo_dir', type=str, default='yolo_labels', 
                       help='Output directory for YOLO txt labels')
    args = parser.parse_args()
    
    convert_json_to_yolo(args.json_dir, args.yolo_dir)