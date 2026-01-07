# Handwriting Recognition

A deep learning system for handwriting recognition using a CRNN (Convolutional Recurrent Neural Network) architecture with CTC (Connectionist Temporal Classification) loss. This model recognizes handwritten text from images using CNNs for feature extraction and bidirectional LSTMs for sequence modeling.

## Features

- **CRNN Architecture**: CNNs extract visual features, bidirectional LSTMs model sequential dependencies
- **CTC Loss**: No character-level alignment required during training
- **PyTorch Implementation**: Built with PyTorch 2.0+ with model compilation support
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training on modern GPUs
- **Gradient Accumulation**: Effective batch sizes larger than GPU memory allows
- **Comprehensive Analysis**: Detailed error analysis with metrics and visualizations
- **Unified CLI**: Single entry point via `main.py` for all operations

## Performance

Based on the full test set analysis (41,370 samples):

| Metric | Result |
|--------|--------|
| **Exact Match Rate** | 79.99% (33,091/41,370) |
| **Character Error Rate (CER)** | 5.50% |
| **Word Error Rate (WER)** | 21.00% |
| **Total Characters** | 270,592 (GT) |
| **Character Errors** | 14,893 |
| **Avg Length Difference** | -0.05 characters |

### Common Error Patterns

The model occasionally confuses:
- Similar-looking characters (e.g., 'I' vs 'l', 'O' vs '0')
- Double letters (e.g., 'EMPTY' → 'EMTY')
- Rare name spellings (e.g., 'LOVIS' → 'LOUIS')

See analysis results in `runs/run_*/analyze/prediction_*_analyze_results.txt` for detailed per-sample predictions.

## Architecture

The model consists of three main components:

### 1. CNN Feature Extractor
```
Input (1×64×256)
  → Conv2D(64) + ReLU + MaxPool2D(2×2)
  → Conv2D(128) + ReLU + MaxPool2D(2×2)
  → Conv2D(256) + ReLU + MaxPool2D(2×2)
  → Features (256×8×32)
```

### 2. Bidirectional LSTM
```
Features reshaped to (batch, 32, 2048)
  → 2-layer Bidirectional LSTM(256)
  → Output (batch, 32, 512)
```

### 3. Fully Connected Classifier
```
LSTM output
  → Linear(512 → num_classes)
  → Log-Softmax
  → CTC predictions
```

### Input/Output Specifications

- **Input**: Grayscale images, size 64×256 pixels
- **Output**: Character sequences from 91-character set (letters, numbers, punctuation, spaces)
- **Character Set**: `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\`"()[]{}<>@#$%^&*+=_/\\|~:;`

## Project Structure

```
handwriting-recognition/
├── data/                           # Dataset directory (created by setup.sh)
│   ├── train/                      # Training images
│   ├── validation/                 # Validation images
│   ├── test/                       # Test images
│   ├── train.csv                   # Training labels (FILENAME, IDENTITY)
│   ├── validation.csv              # Validation labels
│   └── test.csv                    # Test labels
├── runs/                          # Training runs and checkpoints
│   └── run_YYYYMMDD_HHMMSS_checkpoints/  # Timestamped run directories
│       ├── best_model.pth         # Best model (lowest validation CER)
│       ├── epoch_5_model_checkpoint.pth
│       ├── epoch_10_model_checkpoint.pth
│       ├── ...                    # Checkpoints every 5 epochs
│       ├── analyze/               # Analysis results
│       │   └── prediction_YYMMDD_HHMMSS_analyze_results.txt
│       └── plots/                 # Training curves (loss, CER)
├── src/                            # Source code (modular architecture)
│   ├── core/                       # Core utilities (single responsibility modules)
│   │   ├── __init__.py            # Convenience imports
│   │   ├── config.py              # Configuration constants
│   │   ├── checkpoints.py         # Checkpoint loading/finding
│   │   ├── decoding.py            # CTC output decoding
│   │   ├── device.py              # Device management (CPU/GPU)
│   │   ├── evaluation.py          # Model evaluation logic
│   │   ├── metrics.py             # Performance metrics (CER, WER)
│   │   └── training.py            # Training loop logic
│   ├── data/                       # Data pipeline (dataset and transformations)
│   │   ├── __init__.py            # Package exports
│   │   ├── dataset.py             # PyTorch Dataset implementation
│   │   ├── dataloader.py          # DataLoader factory
│   │   ├── transforms.py          # Image preprocessing and augmentation
│   │   └── collate.py             # Batch collation for variable-length sequences
│   ├── models/                     # Model architectures
│   │   ├── __init__.py            # Model exports
│   │   └── crnn.py                # CRNN model (CNN + LSTM + CTC)
│   ├── visualization/              # Plotting and visualization
│   │   ├── __init__.py
│   │   └── plots.py               # Training curves generation
│   ├── train.py                   # Training script (CLI + orchestration)
│   ├── test.py                    # Testing script
│   ├── predict.py                 # Single image prediction (new name)
│   └── analyze.py                 # Comprehensive error analysis
├── main.py                         # Unified CLI interface
├── setup.sh                        # Dataset setup script
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
└── LICENSE                         # MIT License
```

### Key Files

| File | Purpose |
|------|---------|
| [main.py](main.py) | Unified command-line interface for train/test/analyze/generate |
| [src/core/config.py](src/core/config.py) | Central configuration (hyperparameters, paths) |
| [src/core/checkpoints.py](src/core/checkpoints.py) | Checkpoint management (load, find latest) |
| [src/core/decoding.py](src/core/decoding.py) | CTC decoding utilities |
| [src/core/evaluation.py](src/core/evaluation.py) | Model evaluation with CER calculation |
| [src/core/metrics.py](src/core/metrics.py) | Performance metrics (CER, WER, exact match) |
| [src/core/training.py](src/core/training.py) | Training loop with mixed precision |
| [src/models/crnn.py](src/models/crnn.py) | CRNN model definition |
| [src/data/dataset.py](src/data/dataset.py) | PyTorch Dataset for handwriting images |
| [src/data/dataloader.py](src/data/dataloader.py) | DataLoader factory with transforms |
| [src/visualization/plots.py](src/visualization/plots.py) | Training visualization plots |
| [src/train.py](src/train.py) | Training orchestration script |
| [src/test.py](src/test.py) | Model testing and evaluation |
| [src/analyze.py](src/analyze.py) | Detailed error analysis with metrics |
| [src/predict.py](src/predict.py) | Single image inference |

### Module Organization

The codebase follows Kent Beck's refactoring principles with clear separation of concerns:

- **Core modules**: Each handles a single responsibility (device, checkpoints, metrics, etc.)
- **Data pipeline**: Cleanly separated dataset, transforms, and dataloading logic
- **Models**: Self-contained model definitions with unit tests
- **Scripts**: High-level orchestration that composes core functionality
- **No circular dependencies**: Clear import hierarchy (config -> data -> models -> scripts)

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with optional CUDA support)
- torchvision
- NumPy
- Pillow (PIL)
- pandas
- python-Levenshtein
- tqdm
- matplotlib

Install all dependencies:
```bash
pip install torch torchvision numpy pillow pandas python-Levenshtein tqdm matplotlib
```

For GPU acceleration (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow pandas python-Levenshtein tqdm matplotlib
```

## Installation

See [QUICKSTART.md](QUICKSTART.md) for fast setup, or follow these steps:

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd handwriting-recognition
```

2. **Install dependencies** (see Requirements above)

3. **Set up the dataset**:
```bash
bash setup.sh
```

This downloads and organizes the dataset into `data/train/`, `data/validation/`, `data/test/` with corresponding CSV label files.

## Usage

### Training

Train a model from scratch:

```bash
python main.py train
```

**Options:**
- `--train-dir`: Training images directory (default: `./data/train/`)
- `--train-labels`: Training labels CSV (default: `./data/train.csv`)
- `--val-dir`: Validation images directory (default: `./data/validation/`)
- `--val-labels`: Validation labels CSV (default: `./data/validation.csv`)
- `--epochs`: Number of epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--accumulation-steps`: Gradient accumulation steps (default: 4, effective batch = 128)
- `--num-workers`: Data loading workers (default: 4)

**Training process:**
1. Loads training and validation datasets
2. Applies data augmentation (blur, rotation) to training set only
3. Trains with mixed precision (FP16) for speed
4. Validates after each epoch, calculates CER
5. Saves checkpoints every 5 epochs
6. Tracks best model based on validation CER
7. Generates loss/CER plots

**Output:**
- Checkpoints: `runs/run_YYYYMMDD_HHMMSS_checkpoints/`
- Best model: `best_model.pth`
- Plots: `plots/training_loss.png`, `plots/validation_metrics.png`

### Testing

Evaluate the trained model on the test set:

```bash
python main.py test
```

**Options:**
- `--checkpoint`: Path to checkpoint file (optional, auto-detects latest)
- `--test-dir`: Test images directory (default: `./data/test/`)
- `--test-labels`: Test labels CSV (default: `./data/test.csv`)
- `--batch-size`: Batch size (default: 32)
- `--num-workers`: Data loading workers (default: 4)

**Output:**
```
Test Results (from Epoch 30):
Test Loss: 0.0234
Test CER: 5.50%
```

### Analyzing Predictions

Perform comprehensive error analysis:

```bash
# Analyze 100 random samples
python main.py analyze --num-samples 100

# Analyze entire test set
python main.py analyze --all
```

**Options:**
- `--checkpoint`: Path to checkpoint (optional, auto-detects latest)
- `--num-samples`: Number of random samples to analyze (default: 100)
- `--all`: Analyze entire test set instead of random samples
- `--test-dir`: Test images directory
- `--test-labels`: Test labels CSV

**Generates:**
- Console output with metrics and examples
- Detailed file: `runs/.../analyze/prediction_YYMMDD_HHMMSS_analyze_results.txt`

**Analysis includes:**
```
=== prediction_260107_175754 ===

ACCURACY METRICS
Exact Match Rate:        79.99% (33091/41370)
Character Error Rate:    5.50%
Word Error Rate:         21.00%

CHARACTER-LEVEL STATISTICS
Total Characters (GT):   270592
Total Characters (Pred): 268360
Character Errors:        14893
Avg Length Difference:   -0.05 chars

MOST COMMON ERROR PATTERNS
15x: EMPTY... -> EMTY...
12x: EMPTY... -> EMY...
8x: HANON... -> MANON...

DETAILED PREDICTION RESULTS
Sample 1 OK (Edit Distance: 0)
Ground Truth: KEVIN
Prediction:   KEVIN
...
```

### Generating Single Predictions

Predict text from a specific test image:

```bash
python main.py generate --index 0
```

**Options:**
- `--index`: Index of the image in test set (required)
- `--checkpoint`: Path to checkpoint (optional, auto-detects latest)
- `--test-dir`: Test images directory
- `--test-labels`: Test labels CSV

**Example output:**
```
Generation Results (from Epoch 30):
Image index: 0
Predicted text: Maria
Ground truth:   Maria
```

## Configuration

Edit [src/core/config.py](src/core/config.py) to customize training:

```python
# Data paths
TRAIN_DIR = "./data/train/"
VALIDATION_DIR = "./data/validation/"
TEST_DIR = "./data/test/"
TRAIN_LABELS_FILE = "./data/train.csv"
VALIDATION_LABELS_FILE = "./data/validation.csv"
TEST_LABELS_FILE = "./data/test.csv"

# Character set (91 characters)
CHARACTER_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-`\"()[]{}<>@#$%^&*+=_/\\|~:;"

# Training hyperparameters
ACCUMULATION_STEPS = 4      # Gradient accumulation (effective batch = BATCH_SIZE * 4)
BATCH_SIZE = 32             # Batch size per GPU
NUM_WORKERS = 4             # Data loading workers
LEARNING_RATE = 1e-4        # AdamW learning rate
EPOCHS = 30                 # Training epochs

# Image dimensions
IMAGE_HEIGHT = 64           # Input height
IMAGE_WIDTH = 256           # Input width

# Output paths (auto-generated with timestamp)
MODEL_CHECKPOINTS_DIR = f"./runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_checkpoints/"
```

### Module Testing

Each core module includes unit tests that can be run independently:

```bash
# Test dataset implementation
python -m src.data.dataset

# Test dataloader
python -m src.data.dataloader

# Test transforms
python -m src.data.transforms

# Test CRNN model
python -m src.models.crnn
```

All unit tests use ASCII output ("OK" for success) and verify:
- Correct tensor shapes
- Proper data loading
- Model forward pass
- Transform pipelines

### Adjusting for GPU Memory

If you encounter out-of-memory errors:
1. Reduce `BATCH_SIZE` (e.g., 16 or 8)
2. Increase `ACCUMULATION_STEPS` to maintain effective batch size
3. Reduce `NUM_WORKERS` to decrease RAM usage

## Training Pipeline

### Data Preprocessing

1. **Image Loading**: Reads grayscale PNG images
2. **Transformations** (training only):
   - Random Gaussian blur (50% probability)
   - Random rotation (±15 degrees)
3. **Normalization**: Resizes to 64×256, converts to tensor
4. **Label Encoding**: Converts text to character indices using `CHARACTER_SET`

### Training Loop

```
For each epoch:
  1. Train on training set
     - Forward pass with mixed precision (FP16)
     - CTC loss calculation
     - Gradient accumulation over ACCUMULATION_STEPS batches
     - Backward pass and optimizer step

  2. Validate on validation set
     - Forward pass without gradients
     - Calculate validation loss and CER
     - Update best model if CER improved

  3. Save checkpoint every 5 epochs
  4. Generate training plots
```

### Checkpoints

- **`best_model.pth`**: Model with lowest validation CER
- **`epoch_N_model_checkpoint.pth`**: Saved every 5 epochs

Each checkpoint contains:
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': training_loss,
    'val_loss': validation_loss,
    'val_cer': validation_cer
}
```

## Evaluation Metrics

### Character Error Rate (CER)

$$\text{CER} = \frac{\text{Levenshtein Distance}}{\text{Total Characters in Ground Truth}} \times 100\%$$

Measures character-level accuracy. Lower is better. A CER of 5.5% means 94.5% of characters are correct.

### Word Error Rate (WER)

$$\text{WER} = \frac{\text{Word-Level Levenshtein Distance}}{\text{Total Words in Ground Truth}} \times 100\%$$

Measures word-level accuracy. More strict than CER (one character error fails entire word).

### Exact Match Rate

Percentage of predictions that exactly match ground truth (case-sensitive).

## Dataset

Uses the [Handwriting Recognition dataset from Kaggle](https://www.kaggle.com/datasets/landlord/handwriting-recognition).

**Dataset statistics:**
- **Training set**: ~50,000 images
- **Validation set**: ~10,000 images
- **Test set**: 41,370 images
- **Image format**: Grayscale PNG
- **Text content**: Names (first names, last names, full names)
- **Labels**: CSV files with `FILENAME` and `IDENTITY` columns

The setup script (`setup.sh`) downloads and organizes the dataset automatically.

## Advanced Usage

### Using Specific Checkpoints

All commands support `--checkpoint` to specify a model:

```bash
# Test with specific checkpoint
python main.py test --checkpoint ./runs/run_20260107_144221_checkpoints/epoch_20_model_checkpoint.pth

# Analyze with specific checkpoint
python main.py analyze --checkpoint ./runs/run_20260107_144221_checkpoints/best_model.pth --all

# Generate with specific checkpoint
python main.py generate --index 100 --checkpoint ./runs/run_20260107_144221_checkpoints/epoch_15_model_checkpoint.pth
```

### Custom Data Paths

Override default paths:

```bash
# Train with custom paths
python main.py train \
  --train-dir ./custom_data/train/ \
  --train-labels ./custom_data/train.csv \
  --val-dir ./custom_data/val/ \
  --val-labels ./custom_data/val.csv

# Test with custom paths
python main.py test \
  --test-dir ./custom_data/test/ \
  --test-labels ./custom_data/test.csv
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or increase gradient accumulation:
```python
# In config.py
BATCH_SIZE = 16          # Reduced from 32
ACCUMULATION_STEPS = 8   # Increased from 4
```

### Slow Data Loading

Increase workers or enable persistent workers:
```python
# In config.py
NUM_WORKERS = 8  # Increase if you have many CPU cores
```

### Poor Performance

- Train for more epochs
- Reduce learning rate for fine-tuning
- Add more data augmentation
- Check for data quality issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [Handwriting Recognition dataset on Kaggle](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
- **Framework**: PyTorch 2.0+
- **Architecture**: CRNN (Convolutional Recurrent Neural Network) with CTC loss
- **Metrics**: Levenshtein distance (python-Levenshtein)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{handwriting_recognition_2026,
  author = {Miguel Mochizuki Silva},
  title = {Handwriting Recognition with CRNN},
  year = {2026},
  url = {https://github.com/AlbertNewton254/handwriting-recognition}
}
```