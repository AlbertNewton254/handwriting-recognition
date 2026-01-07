"""
training.py
"""

import torch
from torch.cuda.amp import autocast


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
