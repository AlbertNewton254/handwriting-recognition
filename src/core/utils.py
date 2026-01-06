import torch
from .config import CHARACTER_SET


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
    """
    model.eval()
    total_loss = 0.0

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

    return total_loss / len(dataloader)


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