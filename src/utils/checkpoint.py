import torch
import os
import json
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir, 
                   is_best=False, filename=None):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save as best if needed
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved with accuracy: {accuracy:.4f}")
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}, accuracy {accuracy:.4f}")
    
    return epoch, loss, accuracy


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Look for numbered checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    return None


def calculate_accuracy(predictions, targets, target_lengths, blank_idx=0):
    """Calculate sequence accuracy for CTC"""
    correct = 0
    total = 0
    
    # Decode predictions
    for i, (pred, target_len) in enumerate(zip(predictions, target_lengths)):
        # Greedy decode
        pred_indices = torch.argmax(pred, dim=1)
        
        # Remove blanks and consecutive duplicates
        decoded = []
        prev = None
        for idx in pred_indices:
            if idx != blank_idx and idx != prev:
                decoded.append(idx.item())
            prev = idx
        
        # Compare with target
        target_start = sum(target_lengths[:i])
        target_end = target_start + target_len
        target_seq = targets[target_start:target_end].tolist()
        
        if decoded == target_seq:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, log_dir):
    """Log training metrics"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    log_file = os.path.join(log_dir, 'training_log.jsonl')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data) + '\n')


def decode_prediction(prediction, idx_to_char, blank_idx=0):
    """Decode CTC prediction to text"""
    # Greedy decode
    pred_indices = torch.argmax(prediction, dim=1)
    
    # Remove blanks and consecutive duplicates
    decoded = []
    prev = None
    for idx in pred_indices:
        if idx != blank_idx and idx != prev:
            decoded.append(idx_to_char.get(idx.item(), '<UNK>'))
        prev = idx
    
    return ''.join(decoded)