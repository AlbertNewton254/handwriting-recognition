"""
handwriting_dataset.py
"""

import os
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ..core.config import *

class HandwritingDataset(Dataset):
    """
    Dataset for handwriting recognition from images and labels

    images_dir: Directory containing image files
    labels_file: Path to CSV file with image labels
    transform: Optional transform to apply to images
    char2idx: Mapping from characters to indices
    idx2char: Mapping from indices to characters
    num_classes: Total number of character classes
    image_files: Sorted list of image filenames
    labels_dict: Dictionary mapping image IDs to label text
    """
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        labels_df = pd.read_csv(labels_file)
        self.transform = transform

        self.char2idx = {char: idx + 1 for idx, char in enumerate(CHARACTER_SET)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.num_classes = len(self.char2idx) + 1

        # Cache the sorted list of image files (only list directory once!)
        # Sort naturally to match CSV ordering (numeric not lexicographic)
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        self.image_files = sorted(os.listdir(self.images_dir), key=natural_sort_key)

        # Create O(1) lookup dictionary for labels instead of O(n) DataFrame filtering
        self.labels_dict = dict(zip(labels_df['FILENAME'], labels_df['IDENTITY']))

    def __len__(self):
        """
        Get the total number of samples in the dataset

        length: Number of images in the dataset
        """
        return len(self.image_files)

    def text_to_indices(self, text):
        """
        Convert text string to list of character indices

        text: Input text string to convert

        indices: List of integer indices corresponding to characters
        """
        indices = []
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                warnings.warn(f"Character '{char}' not in CHARACTER_SET, skipping", UserWarning)
        return indices

    def indices_to_text(self, indices):
        """
        Convert list of character indices to text string

        indices: List of integer indices to convert

        text: Decoded text string (ignoring padding zeros)
        """
        return ''.join(self.idx2char[idx] for idx in indices if idx != 0)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset

        idx: Index of the sample to retrieve

        image: Transformed image tensor
        label_indices: List of character indices for the image label
        """
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image_id = os.path.splitext(self.image_files[idx])[0] + '.jpg'

        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)

        # O(1) dictionary lookup instead of O(n) DataFrame filtering
        if image_id not in self.labels_dict:
            raise ValueError(f"No label found for image ID: {image_id}")
        label_text = str(self.labels_dict[image_id])
        label_indices = self.text_to_indices(label_text)

        return image, label_indices

if __name__ == "__main__":
    dataset = HandwritingDataset(images_dir=TRAIN_DIR, labels_file=TRAIN_LABELS_FILE)
    sample_image, sample_label = dataset[0]
    print(f"Sample image size: {sample_image.size}")
    print(f"Sample label indices: {sample_label}")
    print(f"Sample label text: {dataset.indices_to_text(sample_label)}")