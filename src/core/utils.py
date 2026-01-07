"""
utils.py

This module re-exports commonly used utility functions for backward compatibility.
"""

# Re-export functions from specialized modules
from .checkpoints import find_latest_checkpoint, load_model_checkpoint
from .decoding import decode_predictions, decode_ground_truth
from .evaluation import evaluate
from .metrics import calculate_cer
from .training import train_one_epoch


def get_device():
    """
    Get the appropriate device for PyTorch operations

    device: 'cuda' if GPU is available, otherwise 'cpu'
    """
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'
