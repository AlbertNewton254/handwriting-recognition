"""
transforms.py

Image transformation pipeline for data augmentation
"""

import torchvision.transforms as T
from ..core.config import *

class HandwritingTransform:
    """
    Transform pipeline for handwriting recognition images

    resize: Target size for resizing images
    random_blur: Whether to apply random Gaussian blur
    rotate: Whether to apply random rotation
    to_tensor: Whether to convert images to tensors
    transform: Composed transformation pipeline
    """
    def __init__(self, resize=(IMAGE_HEIGHT, IMAGE_WIDTH), random_blur=True, rotate=True, to_tensor=True):
        transform_list = []
        if resize:
            transform_list.append(T.Resize(resize))
        if random_blur:
            transform_list.append(T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
        if rotate:
            transform_list.append(T.RandomRotation(degrees=15))
        if to_tensor:
            transform_list.append(T.ToTensor())

        self.transform = T.Compose(transform_list)

    def __call__(self, image):
        """
        Apply the transformation pipeline to an image

        image: Input PIL image

        transformed_image: Transformed image tensor
        """
        return self.transform(image)

if __name__ == "__main__":
    # Unit test for HandwritingTransform
    from PIL import Image
    from ..core.config import IMAGE_HEIGHT, IMAGE_WIDTH

    print("Testing HandwritingTransform...")

    # Test with all transforms
    transform_full = HandwritingTransform(random_blur=True, rotate=True)
    sample_image = Image.new('L', (200, 200), color=255)
    transformed = transform_full(sample_image)
    assert transformed.size(0) == 1, f"Expected 1 channel, got {transformed.size(0)}"
    assert transformed.size(1) == IMAGE_HEIGHT, f"Height mismatch: {transformed.size(1)} != {IMAGE_HEIGHT}"
    assert transformed.size(2) == IMAGE_WIDTH, f"Width mismatch: {transformed.size(2)} != {IMAGE_WIDTH}"
    print(f"OK - Full transform output shape: {transformed.size()}")

    # Test without augmentations
    transform_minimal = HandwritingTransform(random_blur=False, rotate=False)
    transformed_minimal = transform_minimal(sample_image)
    assert transformed_minimal.size() == transformed.size(), "Size mismatch between transform modes"
    print(f"OK - Minimal transform output shape: {transformed_minimal.size()}")

    # Test custom size
    transform_custom = HandwritingTransform(resize=(32, 128))
    transformed_custom = transform_custom(sample_image)
    assert transformed_custom.size() == (1, 32, 128), f"Custom size failed: {transformed_custom.size()}"
    print(f"OK - Custom size transform output shape: {transformed_custom.size()}")

    print("\nAll tests passed!")