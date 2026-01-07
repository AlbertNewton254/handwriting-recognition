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
python main.py train
# or: python -m src.train
```

Trains for 30 epochs, saves checkpoints to `runs/run_YYYYMMDD_HHMMSS_checkpoints/`.

## Test

```bash
python main.py test
# or: python -m src.test
```

Evaluates on test set, shows CER and sample predictions.

## Analyze

```bash
python main.py analyze --num-samples 100
# or: python src.analyze
```

Detailed error analysis. Results saved to `runs/.../analyze/prediction_YYMMDD_HHMMSS_analyze_results.txt`.

Add `--all` flag to analyze entire test set instead of random samples.

## Generate (single image)

```bash
python main.py generate --index 0
# or: python -m src.generate --index 0
```

Predict text from specific test image by index.

## Configuration

Edit [src/core/config.py](src/core/config.py) to adjust:
- `BATCH_SIZE` (default: 32)
- `LEARNING_RATE` (default: 1e-4)
- `EPOCHS` (default: 30)
- `ACCUMULATION_STEPS` (default: 4)
- `IMAGE_HEIGHT/WIDTH` (default: 64×256)

## Commands Summary

| Command | What it does |
|---------|-------------|
| `python main.py train` | Train model from scratch |
| `python main.py test` | Test on test set |
| `python main.py analyze` | Detailed error analysis |
| `python main.py generate --index N` | Predict single image |

All commands support `--checkpoint` to specify a model checkpoint.

## Requirements

- Python 3.8+
- PyTorch + torchvision
- NumPy, Pillow, pandas, python-Levenshtein, tqdm, matplotlib
- (Optional) CUDA-enabled GPU for faster training

For more details, see [README.md](README.md).

## Next Steps

- **Analyze predictions**: Use `python main.py analyze` for detailed error analysis
- **Test individual predictions**: Use `generate.py` to see predictions on different images
- **Experiment with hyperparameters**: Try different learning rates, batch sizes, or model architectures
- **Identify problem areas**: Use the analyze command to examine which characters or words are hardest to recognize
- **Fine-tune the model**: Train for more epochs or adjust the model architecture based on analysis results
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