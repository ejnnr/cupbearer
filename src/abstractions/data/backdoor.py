from dataclasses import dataclass
from typing import Tuple

import numpy as np

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch

from . import DatasetConfig
from ._shared import Transform


class CornerPixelBackdoor(Transform):
    """Adds a white/red pixel to the specified corner of the image and sets the target.

    For grayscale images, the pixel is set to 255 (white),
    for RGB images it is set to (255, 0, 0) (red).

    Args:
        probability: Probability of applying the transform.
        corner: Corner of the image to add the pixel to.
            Can be one of "top-left", "top-right", "bottom-left", "bottom-right".
        target_class: Target class to set the image to after the transform is applied.
    """

    def __init__(
        self,
        p_backdoor: float = 1.0,
        corner="top-left",
        target_class=0,
    ):
        assert 0 <= p_backdoor <= 1, "Probability must be between 0 and 1"
        assert corner in [
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ], "Invalid corner specified"
        self.p_backdoor = p_backdoor
        self.corner = corner
        self.target_class = target_class

    def __call__(self, sample: Tuple[np.ndarray, int]):
        img, target = sample

        # No backdoor, don't do anything
        if torch.rand(1) > self.p_backdoor:
            return img, target

        # Note that channel dimension is last.
        if self.corner == "top-left":
            img[0, 0] = 1
        elif self.corner == "top-right":
            img[-1, 0] = 1
        elif self.corner == "bottom-left":
            img[0, -1] = 1
        elif self.corner == "bottom-right":
            img[-1, -1] = 1

        return img, self.target_class


class NoiseBackdoor(Transform):
    def __init__(
        self, p_backdoor: float = 1.0, std: float = 0.3, target_class: int = 0
    ):
        self.p_backdoor = p_backdoor
        self.std = std
        self.target_class = target_class

    def __call__(self, sample: Tuple[np.ndarray, int]):
        img, target = sample
        if torch.rand(1) > self.p_backdoor:
            return img, target
        else:
            noise = np.random.normal(0, self.std, img.shape)
            img = img + noise
            return img, self.target_class


@dataclass
class BackdoorData(DatasetConfig):
    original: DatasetConfig
    backdoor: Transform

    def __post_init__(self):
        self.transforms = {
            **self.original.transforms,
            **self.transforms,
            "backdoor": self.backdoor,
        }

    def _get_dataset(self):
        return self.original._get_dataset()
