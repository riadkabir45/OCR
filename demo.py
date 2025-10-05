#!/usr/bin/env python3
"""
Demo script to quickly test the word recognition model
"""

import argparse
import os
import sys
import torch
import random
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import WordDataset, get_transforms
from src.models.crnn import create_model
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint, decode_prediction


def demo_single_prediction(model, dataset, device, sample_idx=None):
    """Demo prediction on a single sample"""
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[sample_idx]
    image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
    ground_truth = sample['text']
    
    model.eval()
    with torch.no_grad():
        output = model(image)  # [seq_len, 1, vocab_size]
        prediction = decode_prediction(output[:, 0, :], dataset.idx_to_char, blank_idx=0)
    
    print(f"\nSample {sample_idx}:")
    print(f"Ground Truth: '{ground_truth}'")
    print(f"Prediction:   '{prediction}'")
    print(f"Correct: {'✓' if prediction == ground_truth else '✗'}")
    
    return prediction == ground_truth


def main():
    parser = argparse.ArgumentParser(description='Demo CRNN model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='converted', help='Data directory')
    parser.add_argument('--num_demos', type=int, default=5, help='Number of demo predictions')
    parser.add_argument('--use_best', action='store_true', help='Use best checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    transform = get_transforms(is_training=False)
    dataset = WordDataset(data_dir=args.data_dir, transform=transform)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create model
    vocab_size = len(dataset.char_to_idx)
    model = create_model(vocab_size=vocab_size).to(device)
    
    # Load checkpoint
    if args.use_best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth')
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return
    
    epoch, loss, accuracy = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from epoch {epoch} with accuracy {accuracy:.4f}")
    
    # Demo predictions
    print(f"\nRunning {args.num_demos} demo predictions:")
    correct = 0
    
    for i in range(args.num_demos):
        is_correct = demo_single_prediction(model, dataset, device)
        if is_correct:
            correct += 1
    
    print(f"\nDemo Results: {correct}/{args.num_demos} correct ({correct/args.num_demos*100:.1f}%)")


if __name__ == '__main__':
    main()