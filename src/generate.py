import os
import torch
import argparse
import glob
from src.core.config import TEST_DIR, TEST_LABELS_FILE, CHARACTER_SET
from src.core.utils import decode_predictions
from src.models.handwriting_recognition_model import HandwritingRecognitionModel
from src.data.handwriting_dataloader import get_handwriting_dataloader


def generate_from_model(test_dir, test_labels, checkpoint_path, index, device='cuda'):
    """
    Generate text prediction from a single image using a trained model

    test_dir: Directory containing test images
    test_labels: Path to test labels CSV file
    checkpoint_path: Path to model checkpoint file to load
    index: Index of the image in the dataset
    device: Device to run inference on ('cuda' or 'cpu')

    predicted_text: The predicted text string
    ground_truth_text: The ground truth text string (or None if unavailable)
    """
    # Create test dataloader without augmentation transforms
    test_loader = get_handwriting_dataloader(test_dir, test_labels, batch_size=1, shuffle=False, num_workers=0, with_transform=False)

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

    # Set model to evaluation mode
    model.eval()

    # Get image from dataset
    image_tensor, _ = test_loader.dataset[index]
    image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)

    # Generate prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        # Use decode_predictions to decode the output
        predicted_texts = decode_predictions(outputs, CHARACTER_SET)
        predicted_text = predicted_texts[0]

    # Get ground truth if available
    ground_truth_text = None
    try:
        _, ground_truth_indices = test_loader.dataset[index]
        ground_truth_text = ''.join([CHARACTER_SET[idx - 1] for idx in ground_truth_indices])
    except Exception:
        pass

    print(f"\nGeneration Results{epoch_info}:")
    print(f"Image index: {index}")
    print(f"Predicted text: {predicted_text}")
    if ground_truth_text:
        print(f"Ground truth:   {ground_truth_text}")

    return predicted_text, ground_truth_text


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

    # Generate text from image
    print(f"\nGenerating text for image at index {args.index}...")
    generate_from_model(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        index=args.index,
        device=device
    )
