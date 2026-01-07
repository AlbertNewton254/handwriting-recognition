"""
decoding.py
"""

import torch
from .config import CHARACTER_SET


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
