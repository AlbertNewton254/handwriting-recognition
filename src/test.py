import os
import torch
import torch.nn as nn
import argparse
import glob
from src.core.config import *
from src.core.utils import evaluate
from src.data.handwriting_dataloader import get_handwriting_dataloader
from src.models.handwriting_recognition_model import HandwritingRecognitionModel


def test_model(test_dir, test_labels, checkpoint_path, batch_size=32, num_workers=4, device='cuda'):
    """
    Test a trained handwriting recognition model

    test_dir: Directory containing test images
    test_labels: Path to test labels CSV file
    checkpoint_path: Path to model checkpoint file to load
    batch_size: Batch size for testing
    num_workers: Number of workers for data loading
    device: Device to test on ('cuda' or 'cpu')

    test_loss: Average test loss
    """
    # Create test dataloader without augmentation transforms
    test_loader = get_handwriting_dataloader(test_dir, test_labels, batch_size=batch_size, shuffle=False, num_workers=num_workers, with_transform=False)

    # Initialize model with correct num_classes
    model = HandwritingRecognitionModel(num_classes=test_loader.dataset.num_classes).to(device)

    # Load checkpoint
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

    # Evaluate on test set
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    test_loss, test_cer = evaluate(model, test_loader, criterion=criterion, device=device)

    print(f"\nTest Results{epoch_info}:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test CER: {test_cer:.2f}%")

    return test_loss, test_cer


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