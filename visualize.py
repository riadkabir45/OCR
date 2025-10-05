import matplotlib.pyplot as plt
import random
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import WordDataset, get_transforms


def visualize_samples(dataset, num_samples=8):
    """Visualize random samples from the dataset"""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Get image (without transforms for visualization)
        image = sample['image']
        text = sample['text']
        
        # If image is tensor, convert to numpy
        if hasattr(image, 'numpy'):
            # Denormalize if it was normalized
            if image.shape[0] == 3:  # RGB tensor
                image = image.permute(1, 2, 0)
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f"Text: '{text}'", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('word_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization to word_samples.png")


def show_dataset_stats(dataset):
    """Show dataset statistics"""
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.char_to_idx)}")
    
    # Word length distribution
    lengths = [len(item['text']) for item in dataset.data]
    print(f"Word length range: {min(lengths)} - {max(lengths)}")
    print(f"Average word length: {np.mean(lengths):.2f}")
    
    # Character frequency
    char_count = {}
    for item in dataset.data:
        for char in item['text']:
            char_count[char] = char_count.get(char, 0) + 1
    
    print(f"\nTop 10 most frequent characters:")
    sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:10]:
        print(f"  '{char}': {count}")


if __name__ == "__main__":
    # Create dataset without transforms for visualization
    dataset_vis = WordDataset(transform=None)
    
    # Show stats
    show_dataset_stats(dataset_vis)
    
    # Visualize samples
    visualize_samples(dataset_vis)
    
    # Test with transforms
    print("\nTesting with transforms...")
    dataset_train = WordDataset(transform=get_transforms())
    sample = dataset_train[0]
    print(f"Transformed image shape: {sample['image'].shape}")
    print(f"Text indices shape: {sample['text_indices'].shape}")
    print(f"Sample text: '{sample['text']}'")