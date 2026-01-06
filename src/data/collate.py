import torch

def collate_fn(batch):
    """
    Collate function for batching variable-length sequences

    batch: List of (image, label) tuples

    images: Stacked tensor of images
    labels: Padded tensor of label sequences
    label_lengths: Original lengths of label sequences before padding
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)

    # Store original label lengths before padding
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # Pad variable-length label sequences to the same length
    max_label_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        padded_label = label + [0] * (max_label_len - len(label))
        padded_labels.append(padded_label)

    labels = torch.tensor(padded_labels, dtype=torch.long)
    return images, labels, label_lengths