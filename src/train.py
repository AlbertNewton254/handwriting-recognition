import torch
from src.core.config import *
from src.core.utils import train_model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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