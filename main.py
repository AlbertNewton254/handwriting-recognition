"""
main.py
"""

import argparse
import torch
from src.core.config import (
    TRAIN_DIR, TRAIN_LABELS_FILE,
    VALIDATION_DIR, VALIDATION_LABELS_FILE,
    TEST_DIR, TEST_LABELS_FILE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, ACCUMULATION_STEPS, NUM_WORKERS
)
from src.core.utils import get_device
from src.core.checkpoints import find_latest_checkpoint
from src.train import train_model
from src.test import test_model
from src.generate import generate_from_model
from src.analyze import analyze_predictions


def train_command(args):
    """Execute training command"""
    device = get_device()
    print(f"Using device: {device}")

    train_model(
        train_dir=args.train_dir,
        train_labels=args.train_labels,
        val_dir=args.val_dir,
        val_labels=args.val_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        device=device,
        num_workers=args.num_workers
    )


def test_command(args):
    """Execute testing command"""
    device = get_device()
    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    test_model(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )


def generate_command(args):
    """Execute generation command"""
    device = get_device()
    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    # Generate text from image
    print(f"\nGenerating text for image at index {args.index}...")
    generate_from_model(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        index=args.index,
        device=device
    )


def analyze_command(args):
    """Execute analysis command"""
    device = get_device()

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    analyze_predictions(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        num_samples=args.num_samples,
        use_all=args.all,
        device=device
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Handwriting Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --epochs 30 --batch-size 64

  # Test a model
  python main.py test --checkpoint ./runs/run_20260106_024556_checkpoints/best_model.pth

  # Generate prediction for a single image
  python main.py generate --index 100

  # Analyze predictions on random test samples
  python main.py analyze --num-samples 100
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a handwriting recognition model')
    train_parser.add_argument('--train-dir', type=str, default=TRAIN_DIR,
                             help=f'Directory containing training images (default: {TRAIN_DIR})')
    train_parser.add_argument('--train-labels', type=str, default=TRAIN_LABELS_FILE,
                             help=f'Path to training labels CSV (default: {TRAIN_LABELS_FILE})')
    train_parser.add_argument('--val-dir', type=str, default=VALIDATION_DIR,
                             help=f'Directory containing validation images (default: {VALIDATION_DIR})')
    train_parser.add_argument('--val-labels', type=str, default=VALIDATION_LABELS_FILE,
                             help=f'Path to validation labels CSV (default: {VALIDATION_LABELS_FILE})')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS,
                             help=f'Number of training epochs (default: {EPOCHS})')
    train_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                             help=f'Batch size for training (default: {BATCH_SIZE})')
    train_parser.add_argument('--accumulation-steps', type=int, default=ACCUMULATION_STEPS,
                             help=f'Gradient accumulation steps (default: {ACCUMULATION_STEPS})')
    train_parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                             help=f'Learning rate (default: {LEARNING_RATE})')
    train_parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                             help=f'Number of data loading workers (default: {NUM_WORKERS})')
    train_parser.set_defaults(func=train_command)

    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('--checkpoint', '-c', type=str,
                            help='Path to checkpoint file (uses most recent if not specified)')
    test_parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                            help=f'Directory containing test images (default: {TEST_DIR})')
    test_parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE,
                            help=f'Path to test labels CSV (default: {TEST_LABELS_FILE})')
    test_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                            help=f'Batch size for testing (default: {BATCH_SIZE})')
    test_parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                            help=f'Number of data loading workers (default: {NUM_WORKERS})')
    test_parser.set_defaults(func=test_command)

    # Generate subcommand
    generate_parser = subparsers.add_parser('generate', help='Generate prediction for a single image')
    generate_parser.add_argument('--index', '-i', type=int, required=True,
                                help='Index of the image in the test dataset')
    generate_parser.add_argument('--checkpoint', '-c', type=str,
                                help='Path to checkpoint file (uses most recent if not specified)')
    generate_parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                                help=f'Directory containing test images (default: {TEST_DIR})')
    generate_parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE,
                                help=f'Path to test labels CSV (default: {TEST_LABELS_FILE})')
    generate_parser.set_defaults(func=generate_command)

    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze predictions on random test samples')
    analyze_parser.add_argument('--num-samples', '-n', type=int, default=100,
                               help='Number of random samples to analyze (default: 100, ignored if --all is used)')
    analyze_parser.add_argument('--all', action='store_true',
                               help='Analyze entire test dataset instead of random samples')
    analyze_parser.add_argument('--checkpoint', '-c', type=str,
                               help='Path to checkpoint file (uses most recent if not specified)')
    analyze_parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                               help=f'Directory containing test images (default: {TEST_DIR})')
    analyze_parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE,
                               help=f'Path to test labels CSV (default: {TEST_LABELS_FILE})')
    analyze_parser.set_defaults(func=analyze_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
