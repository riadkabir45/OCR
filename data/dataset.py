import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
 # Removed unused import torch.nn.functional as F
import numpy as np
import cv2

class LocalizationDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.image_files = [f for f in self.files if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Removed unused max_h, max_w, scaling_factor, max_boxes, and cal_max_size()

    def __len__(self):
        return len(self.image_files)

    # Removed cal_max_size() as it is not used

    def __getitem__(self, idx):
        import random
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        image_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(self.img_dir, f"{image_name}.json")
        masks = []
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        H, W = image_np.shape[:2]
        # Generate all masks at full image size
        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x1, y1, x2, y2 = map(int, [round(x1), round(y1), round(x2), round(y2)])
                box_img = image_np[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                box_gray = np.mean(box_img, axis=2).astype(np.uint8) if box_img.ndim == 3 else box_img
                if box_gray.size > 0:
                    box_mask = (box_gray < int(0.01 * 255)).astype(np.uint8)
                    if box_mask.shape[0] > 2 and box_mask.shape[1] > 2:
                        box_mask = cv2.dilate(box_mask, np.ones((3, 3), np.uint8), iterations=1)
                    mask = np.zeros((H, W), dtype=np.uint8)
                    mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = (box_mask > 0).astype(np.uint8)
                    masks.append(mask)
        # Randomly select one quadrant
        quadrant = random.choice([0, 1, 2, 3])
        if quadrant == 0:  # top-left
            y0, y1 = 0, H // 2
            x0, x1 = 0, W // 2
        elif quadrant == 1:  # top-right
            y0, y1 = 0, H // 2
            x0, x1 = W // 2, W
        elif quadrant == 2:  # bottom-left
            y0, y1 = H // 2, H
            x0, x1 = 0, W // 2
        else:  # bottom-right
            y0, y1 = H // 2, H
            x0, x1 = W // 2, W
        # Crop image and masks to the quadrant
        image_np_crop = image_np[y0:y1, x0:x1]
        cropped_masks = [mask[y0:y1, x0:x1] for mask in masks]
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np_crop).permute(2, 0, 1).float()
        resized_masks = [torch.from_numpy(mask) for mask in cropped_masks if np.any(mask)]
        return image_tensor, resized_masks
