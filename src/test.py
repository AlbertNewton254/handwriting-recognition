import os
import torch
import argparse
import glob
from core.config import *
from core.utils import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # If checkpoint is provided, use it
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find all checkpoint directories
        dirs = glob.glob("./runs/run_*_checkpoints")
        if not dirs:
            raise FileNotFoundError("No checkpoint directories found")

        # Sort by timestamp in folder name (most recent first)
        dirs.sort(reverse=True)

        # Use most recent directory
        most_recent_dir = dirs[0]
        checkpoint_path = os.path.join(most_recent_dir, "best_model.pth")

    print(f"Using checkpoint: {checkpoint_path}")

    test_model(
        test_dir=TEST_DIR,
        test_labels=TEST_LABELS_FILE,
        checkpoint_path=checkpoint_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device
    )