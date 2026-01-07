"""
crnn.py - CRNN model for handwriting recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.config import IMAGE_HEIGHT, IMAGE_WIDTH, get_num_classes

class HandwritingRecognitionModel(nn.Module):
    """
    CRNN model for handwriting recognition

    img_height: Height of input images
    num_channels: Number of input channels (1 for grayscale)
    num_classes: Number of output classes (characters + blank token)
    cnn: Convolutional layers for feature extraction
    rnn: Bidirectional LSTM for sequence modeling
    fc: Fully connected layer for classification
    """
    def __init__(self, img_height=IMAGE_HEIGHT, num_channels=1, num_classes=None):
        super(HandwritingRecognitionModel, self).__init__()
        if num_classes is None:
            num_classes = get_num_classes()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(256 * (img_height // 8), 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the model

        x: Input image tensor of shape (batch, channels, height, width)

        output: Log-softmax probabilities of shape (batch, width, num_classes)
        """
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x

if __name__ == "__main__":
    model = HandwritingRecognitionModel()
    sample_input = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # Expected: (1, width, num_classes)