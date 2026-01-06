import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .config import CHARACTER_SET, MODEL_CHECKPOINTS_DIR, MODEL_CHECKPOINT
from data.handwriting_dataloader import get_handwriting_dataloader
from PIL import Image
from data.handwriting_transforms import HandwritingTransform
from models.handwriting_recognition_model import HandwritingRecognitionModel


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

    best_val_loss = float('inf')
    pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")

    for epoch in pbar:
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, accumulation_steps)
        val_loss = evaluate(model, val_dataloader, criterion, device)

        pbar.set_postfix({'Epoch': epoch, 'Train Loss': f"{train_loss:.4f}", 'Val Loss': f"{val_loss:.4f}"})

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

    print(f"\nTraining complete. Checkpoints saved in {MODEL_CHECKPOINTS_DIR}")


def test_model(test_dir, test_labels, checkpoint_path, batch_size=32, num_workers=4, device='cuda'):
    """
    Test a trained handwriting recognition model

    test_dir: Directory containing test images
    test_labels: Path to test labels CSV file
    checkpoint_path: Path to model checkpoint file to load
    batch_size: Batch size for testing
    num_workers: Number of workers for data loading
    device: Device to test on ('cuda' or 'cpu')

    test_loss: Average test loss
    """
    # Create test dataloader without augmentation transforms
    test_loader = get_handwriting_dataloader(test_dir, test_labels, batch_size=batch_size, shuffle=False, num_workers=num_workers, with_transform=False)

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

    # Evaluate on test set
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    test_loss = evaluate(model, test_loader, criterion=criterion, device=device)

    print(f"\nTest Results{epoch_info}:")
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss

def generate_text_from_image(model, test_dir, test_labels, index, device='cuda'):
    """
    Generate text prediction from a single image tensor using the trained model

    model: The trained handwriting recognition model
    image_tensor: Preprocessed image tensor of shape (1, C, H, W)
    device: Device to run inference on ('cuda' or 'cpu')

    predicted_text: The predicted text string
    """
    model.eval()

    test_loader = get_handwriting_dataloader(test_dir, test_labels, batch_size=1, shuffle=False, num_workers=0, with_transform=False)

    image_tensor, _ = test_loader.dataset[index]
    image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)

    predicted_text = ""
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the predicted indices
        predicted_indices = torch.argmax(outputs, dim=2)
        # Collapse repeated characters and remove blanks (assumed to be index 0)
        predicted_indices = predicted_indices.squeeze(0).cpu().numpy()
        collapsed_indices = []
        previous_index = -1
        for index in predicted_indices:
            if index != previous_index and index != 0:
                collapsed_indices.append(index)
            previous_index = index
        # Convert indices to characters
        predicted_text = ''.join([CHARACTER_SET[i - 1] for i in collapsed_indices])  # -1 for zero-based indexing

    return predicted_text