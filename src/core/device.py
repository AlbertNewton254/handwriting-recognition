"""
device.py

Device management utilities for PyTorch operations
"""

import torch


def get_device():
    """
    Get the appropriate device for PyTorch operations

    device: 'cuda' if GPU is available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
