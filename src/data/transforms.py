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
    transform = HandwritingTransform()
    from PIL import Image
    sample_image = Image.new('L', (200, 200), color=255)
    transformed_image = transform(sample_image)
    print(f"Transformed image size: {transformed_image.size()}") # Expected: (1, IMAGE_HEIGHT, IMAGE_WIDTH)