# Handwriting Recognition

A deep learning project for handwriting recognition using a CRNN architecture with CTC loss. This model can recognize handwritten text from images using a combination of convolutional neural networks for feature extraction and bidirectional LSTMs for sequence modeling.

## Features

- **CRNN Architecture**: Combines CNNs for visual feature extraction with bidirectional LSTMs for sequence modeling
- **CTC Loss**: Uses Connectionist Temporal Classification for training without requiring character-level alignment
- **PyTorch Implementation**: Built with PyTorch for flexibility and performance
- **Kaggle Dataset**: Uses the Handwriting Recognition dataset from Kaggle
- **Gradient Accumulation**: Supports gradient accumulation for training with larger effective batch sizes
- **GPU Support**: Automatically detects and uses CUDA if available

## Architecture

The model consists of three main components:

1. **CNN Feature Extractor**: Three convolutional blocks (Conv2D + ReLU + MaxPool2D) that extract visual features from input images
2. **Bidirectional LSTM**: Two-layer bidirectional LSTM that models sequential dependencies in the features
3. **Fully Connected Layer**: Maps LSTM outputs to character probabilities with log-softmax activation

### Input/Output

- **Input**: Grayscale images of size 64×256 pixels
- **Output**: Character sequences from a set of 90+ characters (letters, numbers, punctuation)

## Project Structure

```
handwriting-recognition/
├── data/                           # Dataset directory (created by setup)
│   ├── train/                      # Training images
│   ├── validation/                 # Validation images
│   ├── test/                       # Test images
│   ├── train.csv                   # Training labels
│   ├── validation.csv              # Validation labels
│   └── test.csv                    # Test labels
├── runs/                          # Model checkpoints directory
│   └── run_YYYYMMDD_HHMMSS_checkpoints/  # Timestamped checkpoint directories
│       ├── best_model.pth         # Best model checkpoint
│       ├── epoch_*.pth            # Epoch-specific checkpoints
│       └── plots/                 # Training plots
├── src/                            # Source code
│   ├── core/                       # Core utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration parameters
│   │   └── utils.py               # Training and testing utilities
│   ├── data/                       # Data loading and processing
│   │   ├── __init__.py
│   │   ├── handwriting_dataset.py # PyTorch Dataset implementation
│   │   ├── handwriting_dataloader.py # DataLoader utilities
│   │   ├── handwriting_transforms.py # Image transformations
│   │   └── collate.py             # Custom collate function
│   ├── models/                     # Model architectures
│   │   ├── __init__.py
│   │   └── handwriting_recognition_model.py # CRNN model
│   ├── train.py                   # Training script
│   ├── test.py                    # Testing script
│   ├── generate.py                # Text generation script
│   └── analyze.py                 # Prediction analysis script
├── main.py                         # Unified command-line interface
├── setup.sh                        # Dataset setup script
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
└── LICENSE                         # MIT License
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pillow
- pandas
- python-Levenshtein
- CUDA (optional, for GPU acceleration)

## Installation

See [QUICKSTART.md](QUICKSTART.md) for detailed installation instructions.

## Quick Start

1. **Run the setup script**:
```bash
bash setup.sh
```

2. **Train the model** (using main.py or train.py directly):
```bash
python main.py train
# or
python ./src/train.py
```

3. **Test the model**:
```bash
python main.py test
# or
python ./src/test.py
```

4. **Analyze model predictions** (see detailed error analysis):
```bash
python main.py analyze --num-samples 100
# or
python ./src/analyze.py
```

5. **Generate predictions from individual images**:
```bash
python main.py generate --index 0
# or
python ./src/generate.py --index 0
```

## Configuration

Key parameters can be modified in [src/core/config.py](src/core/config.py):

- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Learning rate for optimizer (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 30)
- `ACCUMULATION_STEPS`: Gradient accumulation steps (default: 4)
- `IMAGE_HEIGHT`: Input image height (default: 64)
- `IMAGE_WIDTH`: Input image width (default: 256)
- `NUM_WORKERS`: Number of data loading workers (default: 4)

## Dataset

This project uses the [Handwriting Recognition dataset from Kaggle](https://www.kaggle.com/datasets/landlord/handwriting-recognition). The dataset contains handwritten text images with corresponding labels.

**Note**: You need to download the dataset manually before running the setup script. See the Installation section for details.

## Training

The training process:

1. Loads images and labels from the training directory
2. Applies transformations (resizing, normalization)
3. Uses CTC loss for sequence alignment
4. Saves checkpoints periodically
5. Validates on the validation set after each epoch

Training logs will show:
- Training loss per batch
- Validation loss and Character Error Rate (CER) per epoch
- Best model checkpoint

## Testing

The testing script:
1. Loads the best model checkpoint
2. Evaluates on the test set
3. Reports Character Error Rate (CER) and other metrics
4. Shows sample predictions

## Analyzing Predictions

The analysis command provides detailed error analysis on random test samples:

```bash
python main.py analyze --num-samples 100
# or
python ./src/analyze.py
```

This will:
1. Load the best model checkpoint (or use `--checkpoint` to specify a different one)
2. Select random samples from the test dataset
3. Generate predictions and compare with ground truth
4. Calculate comprehensive metrics:
   - Exact Match Rate
   - Character Error Rate (CER)
   - Word Error Rate (WER)
   - Character-level and word-level statistics
   - Common error patterns
5. Display correct and incorrect prediction examples
6. Save detailed results to `prediction_analysis_results.txt`

### Options

- `--num-samples`, `-n`: Number of random samples to analyze (default: 100)
- `--checkpoint`, `-c`: Path to a specific checkpoint file (optional, uses best model by default)
- `--test-dir`: Directory containing test images (optional, uses default from config)
- `--test-labels`: Path to test labels CSV file (optional, uses default from config)

### Example Output

```
PREDICTION ANALYSIS RESULTS
======================================================================

Total Samples Analyzed: 100

ACCURACY METRICS
Exact Match Rate:        85.00% (85/100)
Character Error Rate:    3.12%
Word Error Rate:         12.50%

CHARACTER-LEVEL STATISTICS
Total Characters (GT):   850
Total Characters (Pred): 845
Character Errors:        27
Avg Length Difference:   -0.05 chars

MOST COMMON ERROR PATTERNS
5x: MICHEL... -> MICHE...
3x: SOPHIE... -> SOPIE...
...
```

## Generating Predictions

To generate text predictions from individual images:

```bash
python main.py generate --index 0
# or
python ./src/generate.py --index 0
```

This will:
1. Load the best model checkpoint (or use `--checkpoint` to specify a different one)
2. Load the image at the specified index from the test dataset
3. Generate and display the predicted text
4. Show the ground truth for comparison

### Options

- `--index`, `-i`: Index of the image in the test dataset (required)
- `--checkpoint`, `-c`: Path to a specific checkpoint file (optional, uses best model by default)
- `--test-dir`: Directory containing test images (optional, uses default from config)
- `--test-labels`: Path to test labels CSV file (optional, uses default from config)

### Example

```bash
# Generate prediction for image at index 42
python main.py generate --index 42

# Use a specific checkpoint
python main.py generate --index 10 --checkpoint ./runs/run_20260106_224314_checkpoints/epoch_25_model_checkpoint.pth
```

## Model Checkpoints

Model checkpoints are saved in timestamped directories under `runs/`:
- `runs/run_YYYYMMDD_HHMMSS_checkpoints/epoch_X_model_checkpoint.pth`: Checkpoint after epoch X
- `runs/run_YYYYMMDD_HHMMSS_checkpoints/best_model.pth`: Best model based on validation CER

When running test or generate commands without specifying a checkpoint, the script automatically loads the most recent run directory.

## Performance

Model performance depends on:
- Training data quality and quantity
- Hyperparameter tuning
- Training duration
- Hardware (GPU vs CPU)

Expected metrics after training:
- Character Error Rate (CER): Varies depending on dataset and training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Miguel Mochizuki Silva

## Acknowledgments

- Dataset: [Handwriting Recognition dataset on Kaggle](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
- Framework: PyTorch
- Architecture inspiration: CRNN (Convolutional Recurrent Neural Network)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.