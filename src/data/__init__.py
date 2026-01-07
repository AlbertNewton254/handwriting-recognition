"""
__init__.py - Data handling package

This package contains data loading and preprocessing:
- collate: Custom collate function for variable-length sequences
- dataset: HandwritingDataset class
- dataloader: DataLoader factory function
- transforms: Image transformation pipeline
"""

from .collate import collate_fn
from .dataset import HandwritingDataset
from .dataloader import get_handwriting_dataloader
from .transforms import HandwritingTransform
