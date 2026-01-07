"""
dataloader.py

DataLoader factory with configurable transforms and batching
"""

from torch.utils.data import DataLoader
from ..core.config import TRAIN_DIR, TRAIN_LABELS_FILE, BATCH_SIZE, NUM_WORKERS
from .collate import collate_fn
from .dataset import HandwritingDataset
from .transforms import HandwritingTransform

def get_handwriting_dataloader(data_dir, labels_file, batch_size=32, shuffle=True, num_workers=4, with_transform=True):
    """
    Create a dataloader for handwriting recognition dataset

    data_dir: Directory containing image files
    labels_file: Path to CSV file with labels
    batch_size: Batch size for the dataloader
    shuffle: Whether to shuffle the data
    num_workers: Number of workers for data loading
    with_transform: Whether to apply augmentation transforms (training) or just resize (validation)

    dataloader: DataLoader instance configured for handwriting recognition
    """
    transform = HandwritingTransform() if with_transform else HandwritingTransform(random_blur=False, rotate=False)
    dataset = HandwritingDataset(data_dir, labels_file, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches ahead
    )
    return dataloader

if __name__ == "__main__":
    # Unit test for get_handwriting_dataloader
    import torch
    from ..core.config import IMAGE_HEIGHT, IMAGE_WIDTH

    print("Testing get_handwriting_dataloader...")

    # Test with transforms
    dataloader = get_handwriting_dataloader(
        data_dir=TRAIN_DIR,
        labels_file=TRAIN_LABELS_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        with_transform=True
    )
    print(f"OK - DataLoader created with transforms")

    # Test batch retrieval
    images, labels, label_lengths = next(iter(dataloader))
    assert images.size(0) == BATCH_SIZE, f"Batch size mismatch: {images.size(0)} != {BATCH_SIZE}"
    assert images.size(1) == 1, f"Expected 1 channel, got {images.size(1)}"
    assert images.size(2) == IMAGE_HEIGHT, f"Height mismatch: {images.size(2)} != {IMAGE_HEIGHT}"
    assert images.size(3) == IMAGE_WIDTH, f"Width mismatch: {images.size(3)} != {IMAGE_WIDTH}"
    print(f"OK - Batch shape: {images.size()}")
    print(f"OK - Label lengths: {label_lengths.tolist()[:5]}...")

    # Test without transforms
    dataloader_no_transform = get_handwriting_dataloader(
        data_dir=TRAIN_DIR,
        labels_file=TRAIN_LABELS_FILE,
        batch_size=4,
        num_workers=0,
        with_transform=False
    )
    print(f"OK - DataLoader created without transforms")

    print("\nAll tests passed!")