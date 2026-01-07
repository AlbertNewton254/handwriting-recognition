"""
predict.py

Single image text prediction from trained model
"""

import torch
import argparse
from src.core.config import TEST_DIR, TEST_LABELS_FILE, CHARACTER_SET
from src.core.decoding import decode_predictions, decode_ground_truth
from src.core.checkpoints import find_latest_checkpoint, load_model_checkpoint
from src.models.crnn import HandwritingRecognitionModel
from src.data.dataloader import get_handwriting_dataloader


def predict_from_model(test_dir, test_labels, checkpoint_path, index, device='cuda'):
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

    # Initialize and load model
    model = HandwritingRecognitionModel(num_classes=test_loader.dataset.num_classes).to(device)
    model, epoch_info = load_model_checkpoint(model, checkpoint_path, device)
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
        ground_truth_text = decode_ground_truth(ground_truth_indices)
    except Exception:
        pass

    print(f"\nGeneration Results{epoch_info}:")
    print(f"Image index: {index}")
    print(f"Predicted text: {predicted_text}")
    if ground_truth_text:
        print(f"Ground truth:   {ground_truth_text}")

    return predicted_text, ground_truth_text


if __name__ == "__main__":
    from src.core.device import get_device
    parser = argparse.ArgumentParser(description="Generate text prediction from a handwritten image")
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    parser.add_argument('--index', '-i', type=int, required=True, help='Index of the image in the test dataset')
    parser.add_argument('--test-dir', type=str, default=TEST_DIR, help='Directory containing test images')
    parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE, help='Path to test labels CSV file')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)

    print(f"Using checkpoint: {checkpoint_path}")

    # Generate text from image
    print(f"\nGenerating text for image at index {args.index}...")
    predict_from_model(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        index=args.index,
        device=device
    )
