"""
__init__.py - Core utilities package

This package contains core functionality organized by responsibility:
- config: Configuration constants
- checkpoints: Model checkpoint management
- decoding: CTC output decoding
- evaluation: Model evaluation
- metrics: Performance metrics (CER, WER, etc.)
- training: Training loop
- utils: Convenience re-exports for backward compatibility
"""

# Import commonly used items for convenience
from .config import *
from .checkpoints import find_latest_checkpoint, load_model_checkpoint
from .decoding import decode_predictions, decode_ground_truth
from .evaluation import evaluate
from .metrics import calculate_cer, calculate_metrics
from .training import train_one_epoch
from .utils import get_device
