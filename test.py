import argparse
import os
import sys
import torch
import random
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import create_dataloader
from src.models.crnn import create_model
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint, decode_prediction


def test_model(model, dataloader, device, dataset, num_samples=10):
    """Test the model and show predictions"""
    model.eval()
    
    correct = 0
    total = 0
    sample_count = 0
    
    print("Testing model...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch['images'].to(device)
            target_indices = batch['target_indices'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            texts = batch['texts']
            
            # Forward pass
            outputs = model(images)  # [seq_len, batch, vocab_size]
            
            # Process each sample in batch
            batch_size = images.size(0)
            target_start = 0
            
            for i in range(batch_size):
                # Get prediction for this sample
                pred = outputs[:, i, :]  # [seq_len, vocab_size]
                
                # Decode prediction
                pred_text = decode_prediction(pred, dataset.idx_to_char, blank_idx=0)
                
                # Get target text
                target_len = text_lengths[i].item()
                target_end = target_start + target_len
                target_seq = target_indices[target_start:target_end]
                target_text = ''.join([dataset.idx_to_char[idx.item()] for idx in target_seq])
                
                # Check if correct
                if pred_text == target_text:
                    correct += 1
                total += 1
                
                # Show sample predictions
                if sample_count < num_samples:
                    print(f"\nSample {sample_count + 1}:")
                    print(f"  Ground Truth: '{target_text}'")
                    print(f"  Prediction:   '{pred_text}'")
                    print(f"  Correct: {'✓' if pred_text == target_text else '✗'}")
                    sample_count += 1
                
                target_start = target_end
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTest Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test CRNN model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--use_best', action='store_true', help='Use best checkpoint')
    parser.add_argument('--use_latest', action='store_true', help='Use latest checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='converted', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample predictions to show')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataloader, dataset = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        is_training=False
    )
    
    # Create model
    vocab_size = len(dataset.char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    model = create_model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = None
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.use_best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth')
    elif args.use_latest:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    else:
        # Try to find latest by default
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if not checkpoint_path:
            print("No checkpoint found. Please specify --checkpoint, --use_best, or --use_latest")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    epoch, loss, accuracy = load_checkpoint(checkpoint_path, model)
    print(f"Loaded model from epoch {epoch}")
    
    # Test model
    test_accuracy = test_model(model, dataloader, device, dataset, args.num_samples)


if __name__ == '__main__':
    main()