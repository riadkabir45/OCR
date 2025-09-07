import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class LocalizationDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.image_files = [f for f in self.files if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.max_h = 0
        self.max_w = 0
        self.scaling_factor = 0.2
        self.max_boxes = 0
        self.cal_max_size()

    def __len__(self):
        return len(self.image_files)

    def cal_max_size(self):
        for img_file in self.image_files:
            img_path = os.path.join(self.img_dir, img_file)
            image_name = os.path.splitext(img_file)[0]
            json_path = os.path.join(self.img_dir, f"{image_name}.json")
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                with open(json_path, 'r') as f:
                    data = json.load(f)
                boxes = data.get('boxes', [])
                self.max_boxes = max(self.max_boxes, len(boxes))
                self.max_h = max(self.max_h, h)
                self.max_w = max(self.max_w, w)
            except Exception as e:
                continue

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(self.img_dir, f"{image_name}.json")
        masks = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x1, y1, x2, y2 = map(int, [round(x1), round(y1), round(x2), round(y2)])
                    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
                    mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = 1
                    masks.append(torch.from_numpy(mask))
        except Exception as e:
            pass
        return image, masks
