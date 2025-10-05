import os
import glob
import shutil
import random

def organize_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """Organize images and labels into YOLO train/val structure.
    
    Args:
        images_dir: Directory containing source images
        labels_dir: Directory containing YOLO label txt files
        output_dir: Output directory for organized dataset
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(images_dir, '*.png')))
    
    # Shuffle for random split
    random.shuffle(image_files)
    
    # Calculate split point
    split_point = int(len(image_files) * train_ratio)
    train_images = image_files[:split_point]
    val_images = image_files[split_point:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy files to train split
    for img_path in train_images:
        # Copy image
        img_name = os.path.basename(img_path)
        dst_img = os.path.join(output_dir, 'images', 'train', img_name)
        shutil.copy2(img_path, dst_img)
        
        # Copy corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(output_dir, 'labels', 'train', label_name)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"Warning: Label not found for {img_name}")
    
    # Copy files to val split
    for img_path in val_images:
        # Copy image
        img_name = os.path.basename(img_path)
        dst_img = os.path.join(output_dir, 'images', 'val', img_name)
        shutil.copy2(img_path, dst_img)
        
        # Copy corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(output_dir, 'labels', 'val', label_name)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"Warning: Label not found for {img_name}")
    
    print(f"Dataset organized in {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Organize YOLO dataset into train/val splits')
    parser.add_argument('--images_dir', type=str, default='converted', 
                       help='Directory with source images')
    parser.add_argument('--labels_dir', type=str, default='yolo_labels', 
                       help='Directory with YOLO label txt files')
    parser.add_argument('--output_dir', type=str, default='dataset', 
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='Ratio for train split (0.8 = 80% train, 20% val)')
    args = parser.parse_args()
    
    organize_yolo_dataset(args.images_dir, args.labels_dir, args.output_dir, args.train_ratio)