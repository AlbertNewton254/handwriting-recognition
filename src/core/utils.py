import torch
import glob
import os
import sys
import Levenshtein
from .config import CHARACTER_SET
from ..data.handwriting_dataloader import get_handwriting_dataloader


def calculate_cer(predictions, ground_truths):
    """
    Calculate Character Error Rate (CER) for a batch of predictions

    predictions: List of predicted text strings
    ground_truths: List of ground truth text strings

    cer: Character Error Rate as a percentage
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        distance = Levenshtein.distance(pred, truth)
        total_distance += distance
        total_length += len(truth)

    if total_length == 0:
        return 0.0

    return (total_distance / total_length) * 100


def decode_predictions(outputs, char_set):
    """
    Decode CTC outputs to text strings

    outputs: Model outputs of shape (N, T, C)
    char_set: Character set for decoding

    decoded_texts: List of decoded text strings
    """
    decoded_texts = []
    predicted_indices = torch.argmax(outputs, dim=2)

    for sequence in predicted_indices:
        # Collapse repeated characters and remove blanks (index 0)
        collapsed_indices = []
        previous_index = -1
        for idx in sequence.cpu().numpy():
            if idx != previous_index and idx != 0:
                collapsed_indices.append(idx)
            previous_index = idx

        # Convert indices to characters
        text = ''.join([char_set[i - 1] for i in collapsed_indices])
        decoded_texts.append(text)

    return decoded_texts


def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps):
    """
    Train the model for one epoch with gradient accumulation

    model: The model to train
    dataloader: Training data loader
    criterion: Loss function (CTCLoss)
    optimizer: Optimizer
    device: Device to train on ('cuda' or 'cpu')
    accumulation_steps: Number of steps to accumulate gradients

    avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        # Use non_blocking for async GPU transfer
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        outputs = model(images)
        # CTC Loss expects (T, N, C), where T is sequence length, N is batch size, C is num classes
        outputs_permuted = outputs.permute(1, 0, 2)
        # Reuse tensor shape instead of creating new one every batch
        input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

        loss = criterion(outputs_permuted, labels, input_lengths, label_lengths)
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle the case where the last batch is not a multiple of accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data

    model: The model to evaluate
    dataloader: Validation/test data loader
    criterion: Loss function (CTCLoss)
    device: Device to evaluate on ('cuda' or 'cpu')

    avg_loss: Average evaluation loss
    avg_cer: Average Character Error Rate
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_ground_truths = []

    for images, labels, label_lengths in dataloader:
        # Use non_blocking for async GPU transfer
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        outputs = model(images)
        # CTC Loss expects (T, N, C), where T is sequence length, N is batch size, C is num classes
        outputs_permuted = outputs.permute(1, 0, 2)
        input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

        loss = criterion(outputs_permuted, labels, input_lengths, label_lengths)
        total_loss += loss.item()

        # Decode predictions
        predictions = decode_predictions(outputs, CHARACTER_SET)
        all_predictions.extend(predictions)

        # Decode ground truths
        for i, length in enumerate(label_lengths):
            length = length.item()  # Convert tensor to int
            label_indices = labels[i, :length].cpu().tolist()
            ground_truth = ''.join([CHARACTER_SET[idx - 1] for idx in label_indices])
            all_ground_truths.append(ground_truth)

    avg_loss = total_loss / len(dataloader)
    avg_cer = calculate_cer(all_predictions, all_ground_truths)

    return avg_loss, avg_cer


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


def decode_ground_truth(label_indices, char_set=CHARACTER_SET):
    """
    Decode ground truth label indices to text

    label_indices: List or tensor of label indices
    char_set: Character set for decoding

    text: Decoded text string
    """
    if hasattr(label_indices, 'tolist'):
        label_indices = label_indices.tolist()
    return ''.join([char_set[i - 1] for i in label_indices])


def generate_text_from_image(model, test_dir, test_labels, index, device='cuda'):
    """
    Generate text prediction from a single image tensor using the trained model

    model: The trained handwriting recognition model
    test_dir: Directory containing test images
    test_labels: Path to test labels CSV file
    index: Index of the image in the dataset
    device: Device to run inference on ('cuda' or 'cpu')

    predicted_text: The predicted text string
    """
    model.eval()

    test_loader = get_handwriting_dataloader(test_dir, test_labels, batch_size=1, shuffle=False, num_workers=0, with_transform=False)

    image_tensor, _ = test_loader.dataset[index]
    image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(image_tensor)
        # Use decode_predictions to decode the output
        predicted_texts = decode_predictions(outputs, CHARACTER_SET)
        predicted_text = predicted_texts[0]

    return predicted_text