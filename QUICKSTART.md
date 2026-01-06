# Quick Start Guide

This guide will help you get the handwriting recognition model up and running in minutes.

## Prerequisites

- Linux, macOS, or Windows with WSL
- Python 3.8 or higher
- pip (Python package manager)
- curl and unzip utilities
- (Optional) NVIDIA GPU with CUDA support for faster training

## Step-by-Step Installation

### 1. Clone the Repository

If you haven't already, clone or download this repository:

```bash
git clone git@github.com:AlbertNewton254/handwriting-recognition.git
cd handwriting-recognition
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy pillow pandas
```

For CUDA support (GPU acceleration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow pandas
```

### 4. Set Up the Dataset Structure

Run the setup script:

```bash
bash setup.sh
```

This script will:
- Fetch data from `https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition`
- Create the `data/` directory
- Organize training, validation, and test images into separate folders
- Move CSV label files to the correct locations
- Clean up temporary files

After completion, your `data/` directory should look like:
```
data/
├── train/          # Training images
├── validation/     # Validation images
├── test/           # Test images
├── train.csv       # Training labels
├── validation.csv  # Validation labels
└── test.csv        # Test labels
```

### 5. Verify the Setup

Check that the dataset is properly organized:

```bash
ls -lh data/
ls data/train/ | head -5  # Show first 5 training images
```

You should see the three directories and three CSV files.

## Training the Model

### Start Training

Navigate to the source directory and start training:

```bash
python ./src/train.py
```

### What Happens During Training

1. The script detects if CUDA is available and uses GPU if present
2. Loads training and validation data
3. Trains the model for 30 epochs (configurable)
4. Saves checkpoints after each epoch
5. Tracks the best model based on validation CER (Character Error Rate)

### Model Checkpoints

Checkpoints are saved to `model/run_YYYYMMDD_HHMMSS_checkpoints/`:
- `epoch_X_model_checkpoint.pth`: Saved after each epoch
- `best_model.pth`: Best performing model on validation set

## Testing the Model

After training completes, test the model:

```bash
python ./src/test.py
```

This will:
1. Load the best model checkpoint
2. Evaluate on the test set
3. Display metrics (CER, accuracy, etc.)
4. Show sample predictions

## Generating Predictions from Individual Images

To see predictions for specific test images:

```bash
python ./src/generate.py --index 0
```

### Example Output

```
Using device: cuda
Using checkpoint: ./model/run_20260106_024556_checkpoints/best_model.pth
Model loaded successfully (Epoch 30)

Generating text for image at index 0...

Predicted text: Maria
Ground truth:   Maria
```

### Advanced Usage

```bash
# Try different images
python ./src/generate.py --index 5
python ./src/generate.py --index 100

# Use a specific checkpoint
python ./src/generate.py --index 10 --checkpoint ./model/run_20260106_024556_checkpoints/epoch_20_model_checkpoint.pth
```

## Configuration

To modify training parameters, edit [src/core/config.py](src/core/config.py):

```python
# Common parameters to adjust:
BATCH_SIZE = 32 # Reduce if GPU memory is limited
LEARNING_RATE = 1e-4 # Learning rate
EPOCHS = 30 # Number of training epochs
ACCUMULATION_STEPS = 4 # Gradient accumulation (effective batch = 32 * 4 = 128)
NUM_WORKERS = 4 # Data loading workers
```

## Next Steps

- **Test individual predictions**: Use `generate.py` to see predictions on different images
- **Experiment with hyperparameters**: Try different learning rates, batch sizes, or model architectures
- **Analyze results**: Examine which characters or words are hardest to recognize
- **Fine-tune the model**: Train for more epochs or adjust the model architecture
- **Add augmentation**: Implement additional data augmentation techniques
- **Deploy the model**: Use the trained model in a web app or API

## Additional Resources

- Full documentation: See [README.md](README.md)
- PyTorch documentation: https://pytorch.org/docs/
- CTC Loss explained: https://distill.pub/2017/ctc/

## Getting Help

If you encounter issues:

1. Check this guide and [README.md](README.md)
2. Verify your Python and PyTorch versions
3. Check GPU availability with: `python -c "import torch; print(torch.cuda.is_available())"`
4. Open an issue on GitHub with error details

---

**Happy Training!**