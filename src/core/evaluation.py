"""
evaluation.py - Model evaluation functions
"""

import torch
from .config import CHARACTER_SET
from .decoding import decode_predictions
from .metrics import calculate_cer


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
