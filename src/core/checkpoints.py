"""
checkpoints.py
"""

import torch
import glob
import os
import sys


def find_latest_checkpoint(checkpoint_path=None):
    """
    Find the latest checkpoint file

    checkpoint_path: Explicit path to checkpoint (if None, finds most recent)

    path: Path to checkpoint file
    """
    if checkpoint_path is not None:
        return checkpoint_path

    # Find all checkpoint directories
    dirs = glob.glob("./runs/run_*_checkpoints")
    if not dirs:
        print("Error: No checkpoint directories found in ./runs/")
        sys.exit(1)

    # Sort by timestamp in folder name (most recent first)
    dirs.sort(reverse=True)
    most_recent_dir = dirs[0]
    return os.path.join(most_recent_dir, "best_model.pth")


def load_model_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model from checkpoint with proper state_dict handling

    model: Model instance to load weights into
    checkpoint_path: Path to checkpoint file
    device: Device to load model on

    model: Model with loaded weights
    epoch_info: String with epoch information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch_info = f" (Epoch {checkpoint.get('epoch', 'N/A')})"
    else:
        state_dict = checkpoint
        epoch_info = ""

    # Strip _orig_mod. prefix from compiled model if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model, epoch_info
