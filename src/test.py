"""
test.py
"""

import torch
import torch.nn as nn
import argparse
from src.core.config import *
from src.core.utils import evaluate
from src.core.checkpoints import find_latest_checkpoint, load_model_checkpoint
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

    # Initialize and load model
    model = HandwritingRecognitionModel(num_classes=test_loader.dataset.num_classes).to(device)
    model, epoch_info = load_model_checkpoint(model, checkpoint_path, device)

    # Evaluate on test set with CER calculation
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    test_loss, test_cer = evaluate(model, test_loader, criterion=criterion, device=device)

    print(f"\nTest Results{epoch_info}:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test CER: {test_cer:.2f}%")

    return test_loss, test_cer


if __name__ == "__main__":
    from src.core.utils import get_device
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)

    print(f"Using checkpoint: {checkpoint_path}")

    test_model(
        test_dir=TEST_DIR,
        test_labels=TEST_LABELS_FILE,
        checkpoint_path=checkpoint_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device
    )