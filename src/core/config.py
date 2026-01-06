from datetime import datetime

TRAIN_DIR = "./data/train/"
VALIDATION_DIR = "./data/validation/"
TEST_DIR = "./data/test/"

TRAIN_LABELS_FILE = "./data/train.csv"
VALIDATION_LABELS_FILE = "./data/validation.csv"
TEST_LABELS_FILE = "./data/test.csv"

CHARACTER_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-`\"()[]{}<>@#$%^&*+=_/\\|~:;"


ACCUMULATION_STEPS = 4
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 30

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

MODEL_CHECKPOINTS_DIR = f"./runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_checkpoints/"
MODEL_CHECKPOINT = f"{MODEL_CHECKPOINTS_DIR}/epoch_{{epoch}}_model_checkpoint.pth"


def get_num_classes():
    """
    Calculate number of classes including blank token

    num_classes: Total number of classes (CHARACTER_SET length + 1 for blank token)
    """
    return len(CHARACTER_SET) + 1