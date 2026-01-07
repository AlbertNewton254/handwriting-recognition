# Quick Start Guide

Get the handwriting recognition model up and running ASAP.

## Setup (5 minutes)

```bash
# 1. Clone and enter directory
git clone git@github.com:AlbertNewton254/handwriting-recognition.git
cd handwriting-recognition

# 2. Install dependencies
pip install torch torchvision numpy pillow pandas python-Levenshtein tqdm matplotlib

# 3. Setup dataset
bash setup.sh

# Done! Ready to train.
```

## Train

```bash
# Using unified CLI (recommended)
python main.py train

# Or directly
python -m src.train
```

Trains for 30 epochs, saves checkpoints to `runs/run_YYYYMMDD_HHMMSS_checkpoints/`.

**Training features:**
- Mixed precision (FP16) for faster training
- Gradient accumulation (effective batch size: 128)
- Data augmentation (blur, rotation)
- Automatic checkpoint saving every 5 epochs
- Best model tracking based on validation CER
- Training plots (loss, CER curves)

## Test

```bash
# Using unified CLI (recommended)
python main.py test

# Or directly
python -m src.test
```

Evaluates on test set, shows CER and loss metrics.

**Example output:**
```
Using device: cuda
Test Results (Epoch 30):
Test Loss: 0.2409
Test CER: 5.51%
```

## Analyze

```bash
# Analyze 100 random samples
python main.py analyze --num-samples 100

# Analyze entire test set (all 41,370 images)
python main.py analyze --all

# Or directly
python -m src.analyze --num-samples 100
```

Detailed error analysis with metrics. Results saved to `runs/.../analyze/prediction_YYMMDD_HHMMSS_analyze_results.txt`.

**Analysis includes:**
- Exact match rate
- Character Error Rate (CER)
- Word Error Rate (WER)
- Common error patterns
- Sample-by-sample predictions
- Edit distance calculations

## Generate (single image)

```bash
# Using unified CLI (recommended)
python main.py generate --index 0

# Or directly
python -m src.generate --index 0

# Using new predict.py (same functionality)
python -m src.predict --index 0
```

Predict text from specific test image by index.

**Example output:**
```
Generation Results (Epoch 30):
Image index: 0
Predicted text: Maria
Ground truth:   Maria
```

## Configuration

Edit [src/core/config.py](src/core/config.py) to adjust:
- `BATCH_SIZE` (default: 32)
- `LEARNING_RATE` (default: 1e-4)
- `EPOCHS` (default: 30)
- `ACCUMULATION_STEPS` (default: 4, effective batch = 128)
- `IMAGE_HEIGHT/WIDTH` (default: 64x256)

**Pro tip:** The project is now organized with clear separation of concerns:
- `src/core/` - Single-responsibility modules (config, metrics, training, etc.)
- `src/data/` - Dataset and data loading pipeline
- `src/models/` - Model architectures
- `src/visualization/` - Plotting utilities

## Module Testing

Test individual components independently:

```bash
# Test dataset
python -m src.data.dataset
# Output: OK - Dataset loaded, OK - Sample retrieved, etc.

# Test dataloader
python -m src.data.dataloader
# Output: OK - Batch shape verified, OK - Label lengths correct, etc.

# Test transforms
python -m src.data.transforms
# Output: OK - Transform shapes verified

# Test CRNN model
python -m src.models.crnn
# Output: OK - Model forward pass verified
```

All unit tests use ASCII output ("OK" for passing tests, "WRONG" for failures).

## Commands Summary

| Command | What it does |
|---------|-------------|
| `python main.py train` | Train model from scratch (30 epochs) |
| `python main.py test` | Test on test set (41,370 images) |
| `python main.py analyze` | Detailed error analysis |
| `python main.py analyze --all` | Analyze entire test set |
| `python main.py generate --index N` | Predict single image |

All commands support `--checkpoint` to specify a model checkpoint.

**Direct module execution:**
- `python -m src.train` - Same as `python main.py train`
- `python -m src.test` - Same as `python main.py test`
- `python -m src.analyze` - Same as `python main.py analyze`
- `python -m src.predict --index N` - Same as `python main.py generate`

## Requirements

- Python 3.8+
- PyTorch + torchvision
- NumPy, Pillow, pandas, python-Levenshtein, tqdm, matplotlib
- (Optional) CUDA-enabled GPU for faster training

For more details, see [README.md](README.md).

## Next Steps

- **Analyze predictions**: Use `python main.py analyze` for detailed error analysis
- **Test individual predictions**: Use `generate` command to see predictions on different images
- **Experiment with hyperparameters**: Try different learning rates, batch sizes in `src/core/config.py`
- **Identify problem areas**: Use the analyze command to examine which characters or words are hardest to recognize
- **Fine-tune the model**: Train for more epochs or adjust the model architecture
- **Add augmentation**: Modify `src/data/transforms.py` for additional data augmentation
- **Test modules independently**: Run unit tests on dataset, dataloader, transforms, and model
- **Explore the codebase**: Check the modular architecture in `src/core/`, `src/data/`, and `src/models/`

## Project Architecture Highlights

The project follows best practices with:

1. **Modular design**: Each module has a single, clear responsibility
2. **No circular dependencies**: Clean import hierarchy
3. **Unit testable**: Each component can be tested independently
4. **Well-documented**: Uniform docstring style across all files
5. **Separation of concerns**: Core logic separated from CLI/orchestration

**Module breakdown:**
- `src/core/checkpoints.py` - Checkpoint management only
- `src/core/decoding.py` - CTC decoding logic only
- `src/core/device.py` - Device (CPU/GPU) management only
- `src/core/evaluation.py` - Model evaluation only
- `src/core/metrics.py` - Metric calculations only
- `src/core/training.py` - Training loop only

This makes the code easy to understand, test, and extend!

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