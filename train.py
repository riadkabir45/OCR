import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import create_dataloader
from src.models.crnn import create_model
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    calculate_accuracy, log_metrics, decode_prediction
)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, vocab_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch in pbar:
        images = batch['images'].to(device)
        target_indices = batch['target_indices'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Forward pass
        outputs = model(images)  # [seq_len, batch, vocab_size]
        
        # CTC loss expects input_lengths
        seq_len, batch_size = outputs.size(0), outputs.size(1)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Calculate loss
        loss = criterion(outputs, target_indices, input_lengths, text_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            acc = calculate_accuracy(outputs.permute(1, 0, 2), target_indices, text_lengths)
        
        total_loss += loss.item()
        total_accuracy += acc
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['images'].to(device)
            target_indices = batch['target_indices'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # CTC loss
            seq_len, batch_size = outputs.size(0), outputs.size(1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
            
            loss = criterion(outputs, target_indices, input_lengths, text_lengths)
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs.permute(1, 0, 2), target_indices, text_lengths)
            
            total_loss += loss.item()
            total_accuracy += acc
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train CRNN model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='converted', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume_latest', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading dataset...")
    full_dataloader, dataset = create_dataloader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        shuffle=False,
        is_training=True
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=full_dataloader.collate_fn,
        num_workers=0
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=full_dataloader.collate_fn,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    vocab_size = len(dataset.char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    model = create_model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Resume training
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume:
        start_epoch, _, best_accuracy = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    elif args.resume_latest:
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            start_epoch, _, best_accuracy = load_checkpoint(latest_checkpoint, model, optimizer)
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, vocab_size
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_dataloader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log metrics
        log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, args.log_dir)
        
        # Print epoch results
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
        
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc, 
            args.checkpoint_dir, is_best=is_best
        )
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()