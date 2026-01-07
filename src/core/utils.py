"""
utils.py
"""

import torch
from torch.cuda.amp import autocast
from .config import CHARACTER_SET
from .metrics import calculate_cer
from .checkpoints import find_latest_checkpoint, load_model_checkpoint
from ..data.handwriting_dataloader import get_handwriting_dataloader


def get_device():
    """
    Get the appropriate device for PyTorch operations

    device: 'cuda' if GPU is available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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


def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps, scaler=None):
    """
    Train the model for one epoch with gradient accumulation and mixed precision

    model: The model to train
    dataloader: Training data loader
    criterion: Loss function (CTCLoss)
    optimizer: Optimizer
    device: Device to train on ('cuda' or 'cpu')
    accumulation_steps: Number of steps to accumulate gradients
    scaler: GradScaler for mixed precision training (None for CPU)

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

        # Use mixed precision if scaler is available
        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            # CTC Loss expects (T, N, C), where T is sequence length, N is batch size, C is num classes
            outputs_permuted = outputs.permute(1, 0, 2)
            # Reuse tensor shape instead of creating new one every batch
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

            loss = criterion(outputs_permuted, labels, input_lengths, label_lengths)
            loss = loss / accumulation_steps

        # Backward pass with scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle the case where the last batch is not a multiple of accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
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