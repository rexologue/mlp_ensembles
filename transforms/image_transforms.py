import sys
from typing import Union

import cv2
import numpy as np


class Normalize:
    """Transforms image by scaling each pixel to a range [a, b]"""

    def __init__(self, a: Union[float, int] = -1, b: Union[float, int] = 1):
        self.a = a
        self.b = b

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Image normalization process.

        Args:
            img: The image (pixels in [0, 1] range).

        Returns:
            np.ndarray: The normalized image.
        """
        return self.a + (self.b - self.a) * img


class Standardize:
    """Standardizes image with mean and std."""

    def __init__(self, mean: Union[float, list, tuple], std: Union[float, list, tuple]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Image standardization process.

        Args:
            img: The image (pixels in [0, 1] range).

        Returns:
            np.ndarray: The standardized image.
        """
        return (img - self.mean) / self.std


class ToFloat:
    """Converts image from uint to float and scales it to [0, 1] range."""

    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Image dtype converting process.

        Args:
            img: The image.

        Returns:
            np.ndarray: The scaled and converted to float image.
        """
        return img.astype(np.float32) / 255.


class Resize:
    """Image resizing."""

    def __init__(self, size: Union[int, tuple, list]):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Image resizing process.

        Args:
            img: The image.

        Returns:
            np.ndarray: The resized image.
        """
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)


class Sequential:
    """Composes several transforms together."""

    def __init__(self, transforms: list[dict]):
        self.transforms = [
            getattr(sys.modules[__name__], transform['type'].name)(**transform['params']) for transform in transforms
        ]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Applies all transforms consistently."""
        img_aug = img.copy()
        for transform in self.transforms:
            img_aug = transform(img_aug)
            
        return img_aug
