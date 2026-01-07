"""
train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from src.core.config import *
from src.core.utils import train_one_epoch, evaluate
from src.data.dataloader import get_handwriting_dataloader
from src.models.crnn import HandwritingRecognitionModel
from src.visualization.plots import save_loss_plot, save_cer_plot


def train_model(train_dir, train_labels, val_dir, val_labels, num_epochs=10, batch_size=32, accumulation_steps=4, learning_rate=0.001, device='cuda', num_workers=4):
    """
    Train a handwriting recognition model

    train_dir: Directory containing training images
    train_labels: Path to training labels CSV file
    val_dir: Directory containing validation images
    val_labels: Path to validation labels CSV file
    num_epochs: Number of training epochs
    batch_size: Batch size for training
    accumulation_steps: Number of steps to accumulate gradients
    learning_rate: Learning rate for optimizer
    device: Device to train on ('cuda' or 'cpu')
    num_workers: Number of workers for data loading
    """
    # Create checkpoints directory
    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)

    # Create separate dataloaders for training and validation
    # Training data uses transforms, validation uses only resizing
    train_dataloader = get_handwriting_dataloader(train_dir, train_labels, batch_size=batch_size, shuffle=True, num_workers=num_workers, with_transform=True)
    val_dataloader = get_handwriting_dataloader(val_dir, val_labels, batch_size=batch_size, shuffle=False, num_workers=num_workers, with_transform=False)

    # Initialize model with number of classes from training dataset
    model = HandwritingRecognitionModel(num_classes=train_dataloader.dataset.num_classes).to(device)

    # Compile the model for faster training (PyTorch 2.0+)
    model = torch.compile(model)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if device == 'cuda' else None

    best_val_loss = float('inf')

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_cers = []
    epochs_list = []

    pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")

    for epoch in pbar:
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, accumulation_steps, scaler)
        val_loss, val_cer = evaluate(model, val_dataloader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_cers.append(val_cer)
        epochs_list.append(epoch)

        pbar.set_postfix({'Epoch': epoch, 'Train Loss': f"{train_loss:.4f}", 'Val Loss': f"{val_loss:.4f}", 'Val CER': f"{val_cer:.2f}%"})

        # Print CER every 5 epochs for more visibility
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch} - Val CER: {val_cer:.2f}%")

        # At each 5 epochs, save a checkpoint
        if epoch % 5 == 0:
            checkpoint_path = MODEL_CHECKPOINT.format(epoch=epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"\nBest model saved to {best_model_path} (Epoch: {epoch}, Val Loss: {val_loss:.4f})")

    # Create plots directory and generate visualizations
    plots_dir = os.path.join(MODEL_CHECKPOINTS_DIR, "plots")

    loss_plot_path = save_loss_plot(epochs_list, train_losses, val_losses, plots_dir)
    print(f"Loss plot saved to {loss_plot_path}")

    cer_plot_path = save_cer_plot(epochs_list, val_cers, plots_dir)
    print(f"CER plot saved to {cer_plot_path}")

    print(f"\nTraining complete. Checkpoints saved in {MODEL_CHECKPOINTS_DIR}")


if __name__ == "__main__":
    from src.core.utils import get_device
    device = get_device()
    print(f"Using device: {device}")

    train_model(
        train_dir=TRAIN_DIR,
        train_labels=TRAIN_LABELS_FILE,
        val_dir=VALIDATION_DIR,
        val_labels=VALIDATION_LABELS_FILE,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        device=device,
        num_workers=NUM_WORKERS
    )