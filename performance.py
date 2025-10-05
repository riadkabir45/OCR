import argparse
import os
import sys
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import create_dataloader
from src.models.crnn import create_model
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint, decode_prediction


def calculate_detailed_accuracy(model, dataloader, device, dataset):
    """Calculate detailed accuracy metrics"""
    model.eval()
    
    total_samples = 0
    correct_sequences = 0
    correct_characters = 0
    total_characters = 0
    
    word_length_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    character_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    
    predictions = []
    ground_truths = []
    
    print("Evaluating model performance...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
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
                
                predictions.append(pred_text)
                ground_truths.append(target_text)
                
                # Sequence accuracy
                is_correct = pred_text == target_text
                if is_correct:
                    correct_sequences += 1
                total_samples += 1
                
                # Word length accuracy
                word_len = len(target_text)
                word_length_accuracy[word_len]["total"] += 1
                if is_correct:
                    word_length_accuracy[word_len]["correct"] += 1
                
                # Character-level accuracy
                pred_chars = list(pred_text)
                target_chars = list(target_text)
                
                # Calculate character accuracy using edit distance approach
                max_len = max(len(pred_chars), len(target_chars))
                for j in range(max_len):
                    if j < len(target_chars):
                        char = target_chars[j]
                        character_accuracy[char]["total"] += 1
                        if j < len(pred_chars) and pred_chars[j] == char:
                            character_accuracy[char]["correct"] += 1
                            correct_characters += 1
                        total_characters += 1
                
                target_start = target_end
    
    # Calculate metrics
    sequence_accuracy = correct_sequences / total_samples if total_samples > 0 else 0.0
    character_accuracy_overall = correct_characters / total_characters if total_characters > 0 else 0.0
    
    return {
        'sequence_accuracy': sequence_accuracy,
        'character_accuracy': character_accuracy_overall,
        'total_samples': total_samples,
        'correct_sequences': correct_sequences,
        'word_length_accuracy': dict(word_length_accuracy),
        'character_accuracy_by_char': dict(character_accuracy),
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def show_sample_predictions(predictions, ground_truths, num_samples=20):
    """Show sample predictions vs ground truth"""
    print(f"\n{'='*60}")
    print(f"SAMPLE PREDICTIONS (showing {num_samples} examples)")
    print(f"{'='*60}")
    
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    correct = 0
    for i, idx in enumerate(indices):
        pred = predictions[idx]
        truth = ground_truths[idx]
        is_correct = pred == truth
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n{i+1:2d}. {status} Ground Truth: '{truth}'")
        print(f"    Prediction:   '{pred}'")
    
    print(f"\nSample accuracy: {correct}/{len(indices)} ({correct/len(indices)*100:.1f}%)")


def plot_performance_charts(results, save_dir="performance_charts"):
    """Create performance visualization charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Word length vs accuracy
    word_lengths = sorted(results['word_length_accuracy'].keys())
    accuracies = []
    sample_counts = []
    
    for length in word_lengths:
        data = results['word_length_accuracy'][length]
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        accuracies.append(acc)
        sample_counts.append(data['total'])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(word_lengths, accuracies, alpha=0.7, color='skyblue')
    plt.xlabel('Word Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Word Length')
    plt.grid(True, alpha=0.3)
    
    # Add sample count labels
    for i, (length, acc, count) in enumerate(zip(word_lengths, accuracies, sample_counts)):
        plt.text(length, acc + 0.01, f'n={count}', ha='center', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.bar(word_lengths, sample_counts, alpha=0.7, color='lightcoral')
    plt.xlabel('Word Length')
    plt.ylabel('Sample Count')
    plt.title('Sample Distribution by Word Length')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'word_length_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Character accuracy (top 20 most frequent)
    char_data = results['character_accuracy_by_char']
    sorted_chars = sorted(char_data.items(), key=lambda x: x[1]['total'], reverse=True)[:20]
    
    chars = [item[0] for item in sorted_chars]
    char_accuracies = []
    char_counts = []
    
    for char, data in sorted_chars:
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        char_accuracies.append(acc)
        char_counts.append(data['total'])
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(chars)), char_accuracies, alpha=0.7, color='lightgreen')
    plt.xlabel('Characters')
    plt.ylabel('Accuracy')
    plt.title('Character-level Accuracy (Top 20 Most Frequent)')
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add count labels
    for i, (acc, count) in enumerate(zip(char_accuracies, char_counts)):
        plt.text(i, acc + 0.01, f'n={count}', ha='center', fontsize=8, rotation=90)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(chars)), char_counts, alpha=0.7, color='orange')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Character Frequency (Top 20)')
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'character_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Performance charts saved to '{save_dir}' directory")


def print_performance_summary(results, model_info):
    """Print detailed performance summary"""
    print(f"\n{'='*80}")
    print(f"MODEL PERFORMANCE REPORT")
    print(f"{'='*80}")
    
    print(f"\nModel Information:")
    print(f"  Epoch: {model_info['epoch']}")
    print(f"  Training Loss: {model_info['loss']:.4f}")
    print(f"  Validation Accuracy: {model_info['accuracy']:.4f}")
    
    print(f"\nOverall Performance:")
    print(f"  Total Samples: {results['total_samples']:,}")
    print(f"  Sequence Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_accuracy']*100:.2f}%)")
    print(f"  Character Accuracy: {results['character_accuracy']:.4f} ({results['character_accuracy']*100:.2f}%)")
    print(f"  Correct Sequences: {results['correct_sequences']:,}/{results['total_samples']:,}")
    
    # Word length breakdown
    print(f"\nAccuracy by Word Length:")
    word_length_acc = results['word_length_accuracy']
    for length in sorted(word_length_acc.keys()):
        data = word_length_acc[length]
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"  Length {length:2d}: {acc:.3f} ({data['correct']:3d}/{data['total']:3d} samples)")
    
    # Character accuracy summary
    char_data = results['character_accuracy_by_char']
    if char_data:
        print(f"\nTop 10 Most Accurate Characters:")
        sorted_by_acc = sorted(char_data.items(), 
                              key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                              reverse=True)[:10]
        
        for char, data in sorted_by_acc:
            if data['total'] >= 10:  # Only show characters with enough samples
                acc = data['correct'] / data['total']
                print(f"  '{char}': {acc:.3f} ({data['correct']}/{data['total']})")


def save_performance_report(results, model_info, output_file="performance_report.json"):
    """Save performance report to JSON file"""
    report = {
        'model_info': model_info,
        'performance': {
            'sequence_accuracy': results['sequence_accuracy'],
            'character_accuracy': results['character_accuracy'],
            'total_samples': results['total_samples'],
            'correct_sequences': results['correct_sequences']
        },
        'word_length_accuracy': results['word_length_accuracy'],
        'character_accuracy_by_char': results['character_accuracy_by_char']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CRNN model performance')
    parser.add_argument('--use_best', action='store_true', help='Use best checkpoint instead of latest')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='converted', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of sample predictions to show')
    parser.add_argument('--save_charts', action='store_true', help='Save performance charts')
    parser.add_argument('--output_dir', type=str, default='performance_analysis', help='Output directory for charts and reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    model = create_model(vocab_size=vocab_size).to(device)
    
    # Load checkpoint
    checkpoint_path = None
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.use_best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth')
        print("Using BEST model for evaluation")
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        print("Using LATEST model for evaluation")
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    epoch, loss, accuracy = load_checkpoint(checkpoint_path, model)
    model_info = {'epoch': epoch, 'loss': loss, 'accuracy': accuracy}
    
    # Evaluate model
    results = calculate_detailed_accuracy(model, dataloader, device, dataset)
    
    # Print results
    print_performance_summary(results, model_info)
    
    # Show sample predictions
    show_sample_predictions(results['predictions'], results['ground_truths'], args.num_samples)
    
    # Save performance charts
    if args.save_charts:
        chart_dir = os.path.join(args.output_dir, 'charts')
        plot_performance_charts(results, chart_dir)
    
    # Save detailed report
    report_file = os.path.join(args.output_dir, 'performance_report.json')
    save_performance_report(results, model_info, report_file)


if __name__ == '__main__':
    main()