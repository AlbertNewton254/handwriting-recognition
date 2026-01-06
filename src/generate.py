import os
import torch
import argparse
import glob
from src.core.config import TEST_DIR, TEST_LABELS_FILE
from src.core.utils import generate_text_from_image
from src.models.handwriting_recognition_model import HandwritingRecognitionModel
from src.data.handwriting_dataloader import get_handwriting_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text prediction from a handwritten image")
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    parser.add_argument('--index', '-i', type=int, required=True, help='Index of the image in the test dataset')
    parser.add_argument('--test-dir', type=str, default=TEST_DIR, help='Directory containing test images')
    parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE, help='Path to test labels CSV file')
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
            raise FileNotFoundError("No checkpoint directories found in ./runs/")

        # Sort by timestamp in folder name (most recent first)
        dirs.sort(reverse=True)

        # Use most recent directory
        most_recent_dir = dirs[0]
        checkpoint_path = os.path.join(most_recent_dir, "best_model.pth")

    print(f"Using checkpoint: {checkpoint_path}")

    # Load the model
    test_loader = get_handwriting_dataloader(args.test_dir, args.test_labels, batch_size=1, shuffle=False, num_workers=0, with_transform=False)
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

    print(f"Model loaded successfully{epoch_info}")

    # Generate text from image
    print(f"\nGenerating text for image at index {args.index}...")
    predicted_text = generate_text_from_image(
        model=model,
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        index=args.index,
        device=device
    )

    print(f"\nPredicted text: {predicted_text}")

    # Optionally show the ground truth if available
    try:
        _, ground_truth = test_loader.dataset[args.index]
        ground_truth_text = ''.join([test_loader.dataset.idx2char[idx] for idx in ground_truth])
        print(f"Ground truth:   {ground_truth_text}")
    except Exception as e:
        print(f"Could not retrieve ground truth: {e}")
