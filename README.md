# Bengali Word Recognition with CNN-RNN-CTC# Bengali Word Dataset for Character Recognition



A PyTorch implementation of a hybrid CNN-RNN-CTC model for Bengali word recognition from handwritten text images.A PyTorch-based dataset and visualization toolkit for Bengali word character recognition from images with bounding box annotations.



## Project Structure## Features



```- 🖼️ **Word-level image cropping** from annotated images

HCRC/- 📝 **Character-level vocabulary** creation and encoding

├── src/- 🔄 **Data augmentation** and preprocessing pipelines

│   ├── data/- 📊 **Comprehensive visualization** tools

│   │   ├── __init__.py- 🤖 **PyTorch integration** with custom dataset and data loaders

│   │   └── dataset.py          # Dataset implementation- 📈 **Interactive analysis** and statistics

│   ├── models/- 🏗️ **Model training** examples

│   │   ├── __init__.py

│   │   └── crnn.py             # CNN-RNN-CTC model## Dataset Structure

│   ├── utils/

│   │   ├── __init__.pyThe dataset expects images and corresponding JSON annotation files in the following format:

│   │   └── checkpoint.py       # Training utilities

│   └── __init__.py```

├── converted/                  # Your image and JSON dataconverted/

├── checkpoints/               # Model checkpoints├── 0_21_0.jpg          # Image file

├── logs/                      # Training logs├── 0_21_0.json         # Annotation file

├── train.py                   # Training script├── 1_20_0.jpg

├── test.py                    # Testing script├── 1_20_0.json

├── demo.py                    # Quick demo script└── ...

├── visualize.py              # Data visualization```

└── requirements.txt

```### Annotation Format



## FeaturesEach JSON file contains word-level annotations:



- **CNN-RNN-CTC Architecture**: Hybrid model combining CNN feature extraction, bidirectional LSTM, and CTC loss```json

- **Checkpoint Management**: Automatic saving of best and latest checkpoints{

- **Resume Training**: Continue training from any checkpoint  "shapes": [

- **Data Augmentation**: Color jitter and normalization for better generalization    {

- **Progress Tracking**: tqdm progress bars and detailed logging      "label": "বৃহত্তর",           // Bengali word text

- **Comprehensive Testing**: Evaluation with sample predictions      "points": [                   // Bounding box coordinates

        [28.93, 88.29],            // Top-left point [x, y]

## Dataset        [175.36, 165.07]           // Bottom-right point [x, y]

      ]

The dataset contains Bengali word images with corresponding JSON annotations:    }

- **21,233 word samples** from handwritten text  ],

- **126 unique characters** in vocabulary  "imagePath": "0_21_0.jpg",       // Corresponding image file

- **Bounding box annotations** for precise word cropping  "imageHeight": 1179,

- **CTC-compatible formatting** for sequence learning  "imageWidth": 1448

}

## Model Architecture```



- **CNN Backbone**: 5 convolutional blocks with batch normalization## Installation

- **Feature Mapping**: Linear projection to RNN input space

- **Bidirectional LSTM**: 2-layer bidirectional LSTM (512 hidden units)1. **Clone or download** this repository

- **CTC Output**: Character-level classification with CTC loss2. **Install required packages**:



## Usage```bash

pip install torch torchvision pillow matplotlib numpy opencv-python tqdm scikit-learn

### Training```



```bash## Quick Start

# Basic training

python train.py### 1. Basic Usage



# Custom parameters```python

python train.py --epochs 50 --batch_size 64 --lr 0.001from word_dataset import WordCropDataset

from data_utils import create_enhanced_transforms

# Resume from latest checkpoint

python train.py --resume_latest# Create transforms

transforms = create_enhanced_transforms(image_size=(64, 256), augment=True)

# Resume from specific checkpoint

python train.py --resume checkpoints/checkpoint_epoch_10.pth# Load dataset

```dataset = WordCropDataset(

    data_dir="converted",

### Testing    transform=transforms,

    min_word_length=1,

```bash    max_word_length=20

# Test with best model)

python test.py --use_best

print(f"Dataset loaded with {len(dataset)} samples")

# Test with latest checkpointprint(f"Vocabulary size: {dataset.vocab_size}")

python test.py --use_latest

# Get a sample

# Test with specific checkpointsample = dataset[0]

python test.py --checkpoint checkpoints/best.pthprint(f"Image shape: {sample['image'].shape}")

```print(f"Label: {sample['label']}")

print(f"Encoded label: {sample['encoded_label']}")

### Quick Demo```



```bash### 2. Data Loaders

# Run demo with 5 sample predictions

python demo.py```python

from data_utils import DatasetManager

# Use best model for demo

python demo.py --use_best --num_demos 10# Create data manager with train/val/test splits

```data_manager = DatasetManager(

    dataset=dataset,

### Visualize Data    train_ratio=0.8,

    val_ratio=0.1,

```bash    test_ratio=0.1,

# Show dataset samples and statistics    batch_size=32

python visualize.py)

```

# Get data loaders

## Training Argumentstrain_loader = data_manager.train_loader

val_loader = data_manager.val_loader

```bashtest_loader = data_manager.test_loader

python train.py --help

```# Iterate through batches

for batch in train_loader:

Key arguments:    images = batch['images']        # Shape: (batch_size, 3, 64, 256)

- `--epochs`: Number of training epochs (default: 100)    labels = batch['labels']        # List of strings

- `--batch_size`: Batch size (default: 32)    encoded = batch['encoded_labels']  # Shape: (batch_size, max_length)

- `--lr`: Learning rate (default: 0.001)    break

- `--resume_latest`: Resume from latest checkpoint```

- `--resume`: Resume from specific checkpoint

- `--hidden_size`: LSTM hidden size (default: 256)### 3. Visualization

- `--num_layers`: Number of LSTM layers (default: 2)

- `--dropout`: Dropout rate (default: 0.1)```python

from visualizer import WordDatasetVisualizer

## Test Arguments

# Create visualizer

```bashvisualizer = WordDatasetVisualizer(dataset)

python test.py --help

```# Show random samples

visualizer.visualize_random_samples(num_samples=12)

Key arguments:

- `--use_best`: Use best checkpoint# Show character distribution

- `--use_latest`: Use latest checkpointvisualizer.visualize_character_distribution(top_n=30)

- `--checkpoint`: Specific checkpoint path

- `--num_samples`: Number of sample predictions to show# Show word length distribution

visualizer.visualize_word_length_distribution()

## Installation

# Show full images with annotations

```bashvisualizer.visualize_full_image_annotations(num_images=3)

pip install -r requirements.txt```

```

## Running the Demo

## Model Performance

### Interactive Demo

The model uses CTC loss for sequence-to-sequence learning without requiring character-level alignment. Performance metrics include:

- **Character-level accuracy**```bash

- **Sequence-level accuracy**python main.py

- **Training/validation loss tracking**```



Checkpoints are automatically saved for:This will start an interactive demo with the following options:

- **Best model** (highest validation accuracy)- Load and analyze the dataset

- **Latest model** (most recent epoch)- View random word samples

- **Regular intervals** (every epoch)- Explore character distributions

- Analyze word lengths

## Notes- Save visualizations



- Model automatically handles variable-length sequences### Quick Demo

- CTC decoding removes blanks and consecutive duplicates

- Supports CUDA for GPU acceleration```bash

- Includes gradient clipping for stable trainingpython main.py --quick

- Learning rate scheduling with ReduceLROnPlateau```

### Limited Samples (for testing)

```bash
python main.py --max-samples 1000
```

## Dataset Classes

### WordCropDataset

Main dataset class that handles:
- Loading images and annotations
- Cropping word regions from images
- Creating character vocabulary
- Encoding/decoding text

**Key Methods:**
- `encode_text(text)`: Convert text to character indices
- `decode_text(indices)`: Convert indices back to text
- `get_statistics()`: Get dataset statistics

### DatasetManager

Manages dataset splits and data loaders:
- Automatic train/validation/test splits
- Configurable batch sizes
- Custom collate function for variable-length sequences

### ImageTransforms

Provides image preprocessing pipelines:
- Data augmentation for training
- Normalization and resizing
- Aspect ratio preservation

## Visualization Tools

### WordDatasetVisualizer

Comprehensive visualization toolkit:

1. **Random Samples**: Show cropped word images with labels
2. **Full Image Annotations**: Display original images with bounding boxes
3. **Character Distribution**: Bar charts of character frequencies
4. **Word Length Analysis**: Histograms and statistics
5. **Vocabulary Overview**: Complete vocabulary information
6. **Batch Visualization**: Show processed batches

### Interactive Features

- Menu-driven interface
- Customizable parameters
- Save visualizations to files
- Real-time statistics

## Model Training Example

```python
from simple_model import SimpleCNN, BengaliWordRecognizer

# Create model
model = SimpleCNN(vocab_size=dataset.vocab_size, max_length=20)

# Create recognizer
recognizer = BengaliWordRecognizer(model, dataset, device='cpu')

# Train model
recognizer.train(
    train_loader=data_manager.train_loader,
    val_loader=data_manager.val_loader,
    num_epochs=10,
    learning_rate=0.001
)

# Make predictions
predicted_text = recognizer.predict(image_tensor)
```

## Dataset Statistics

The dataset provides comprehensive statistics:

- **Total samples**: Number of word images
- **Vocabulary size**: Unique characters
- **Word length distribution**: Min, max, average lengths
- **Character frequencies**: Most common characters
- **Image dimensions**: Original and processed sizes

## File Structure

```
├── word_dataset.py      # Main dataset class
├── data_utils.py        # Data processing utilities
├── visualizer.py        # Visualization tools
├── simple_model.py      # Example CNN model
├── main.py             # Demo script
├── README.md           # This file
└── converted/          # Your data directory
    ├── *.jpg           # Image files
    └── *.json          # Annotation files
```

## Configuration Options

### Dataset Parameters

- `data_dir`: Path to data directory
- `transform`: Image transformations
- `max_samples`: Limit dataset size (for testing)
- `min_word_length`: Filter short words
- `max_word_length`: Filter long words

### Image Processing

- `image_size`: Target image dimensions (height, width)
- `augment`: Enable/disable data augmentation
- `normalize`: Color normalization parameters

### Data Loading

- `batch_size`: Batch size for training
- `num_workers`: Parallel data loading workers
- `train_ratio`: Proportion for training set
- `val_ratio`: Proportion for validation set
- `test_ratio`: Proportion for test set

## Advanced Usage

### Custom Transforms

```python
import torchvision.transforms as transforms

custom_transforms = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = WordCropDataset(data_dir="converted", transform=custom_transforms)
```

### Character Filtering

```python
# Create vocabulary with specific characters only
dataset = WordCropDataset(
    data_dir="converted",
    min_word_length=3,      # At least 3 characters
    max_word_length=15      # At most 15 characters
)
```

### Batch Processing

```python
from word_dataset import collate_fn
from torch.utils.data import DataLoader

# Custom data loader
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)
```

## Performance Tips

1. **Use appropriate batch sizes** based on your GPU memory
2. **Set num_workers=0** on Windows to avoid multiprocessing issues
3. **Limit max_samples** during development for faster iteration
4. **Use data augmentation** only for training set
5. **Monitor memory usage** with large datasets

## Troubleshooting

### Common Issues

1. **"No data files found"**
   - Ensure the `converted` directory exists
   - Check that JSON and image files are present

2. **Memory errors**
   - Reduce batch size
   - Set `max_samples` to a smaller number
   - Use fewer data loading workers

3. **Slow loading**
   - Use SSD storage for data
   - Increase `num_workers` (not on Windows)
   - Consider data preprocessing

4. **Visualization issues**
   - Install matplotlib backend: `pip install PyQt5`
   - For headless servers, save visualizations instead of showing

### Windows Specific

- Set `num_workers=0` in DataLoader
- Use forward slashes in paths or raw strings
- Install Visual C++ redistributables for OpenCV

## Contributing

Feel free to contribute by:
- Adding new visualization features
- Implementing advanced models
- Improving data preprocessing
- Adding more evaluation metrics
- Creating documentation

## License

This project is open source. Please cite appropriately if used in research.

## Contact

For questions or issues, please create an issue in the repository or contact the maintainers.