import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class WordDataset(Dataset):
    def __init__(self, data_dir='converted', transform=None, max_length=20):
        """
        Simple word dataset for character recognition
        
        Args:
            data_dir: Directory containing image and json files
            transform: Image transformations
            max_length: Maximum word length for padding
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data()
        
        # Create character vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocab()
        
    def _load_data(self):
        """Load all word samples from JSON files"""
        data = []
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(self.data_dir, json_file)
            
            # Get corresponding image file
            img_file = json_file.replace('.json', '.jpg')
            img_path = os.path.join(self.data_dir, img_file)
            
            # Check if image file exists
            if not os.path.exists(img_path):
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Extract word samples from annotations
                for shape in annotations['shapes']:
                    text = shape['label'].strip()
                    points = shape['points']
                    
                    if text and len(text) <= self.max_length and len(points) == 2:
                        # Extract bounding box
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                        
                        data.append({
                            'image_path': img_path,
                            'bbox': bbox,
                            'text': text
                        })
                        
            except Exception as e:
                continue
        
        print(f"Loaded {len(data)} word samples")
        return data
    
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = set()
        for item in self.data:
            chars.update(item['text'])
        
        # Add special tokens (blank for CTC)
        chars.add('<BLANK>')  # CTC blank token
        chars.add('<UNK>')    # Unknown token
        
        # Create mappings (blank should be index 0 for CTC)
        char_list = ['<BLANK>'] + sorted(list(chars - {'<BLANK>'}))
        char_to_idx = {char: idx for idx, char in enumerate(char_list)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    def _text_to_indices(self, text):
        """Convert text to indices (without blank tokens for CTC)"""
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and crop image
        image = Image.open(item['image_path']).convert('RGB')
        bbox = item['bbox']
        cropped_image = image.crop(bbox)
        
        # Apply transforms
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # Convert text to indices
        text_indices = self._text_to_indices(item['text'])
        
        return {
            'image': cropped_image,
            'text': item['text'],
            'text_indices': text_indices,
            'text_length': len(item['text'])
        }


def get_transforms(image_size=(64, 256), is_training=True):
    """Get image transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """Custom collate function for CTC loss"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    text_indices = [item['text_indices'] for item in batch]
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
    
    # Concatenate all text indices for CTC loss
    target_indices = torch.cat(text_indices, dim=0)
    
    return {
        'images': images,
        'texts': texts,
        'target_indices': target_indices,
        'text_lengths': text_lengths
    }


def create_dataloader(data_dir='converted', batch_size=32, shuffle=True, is_training=True):
    """Create DataLoader"""
    transform = get_transforms(is_training=is_training)
    dataset = WordDataset(data_dir=data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset