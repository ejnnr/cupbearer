from dataclasses import dataclass
from typing import Tuple

import numpy as np

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch

from ._shared import Transform


@dataclass
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

    p_backdoor: float = 1.0
    corner: str = "top-left"
    target_class: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"
        assert self.corner in [
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ], "Invalid corner specified"

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


@dataclass
class NoiseBackdoor(Transform):
    p_backdoor: float = 1.0
    std: float = 0.3
    target_class: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"

    def __call__(self, sample: Tuple[np.ndarray, int]):
        img, target = sample
        if torch.rand(1) > self.p_backdoor:
            return img, target
        else:
            noise = np.random.normal(0, self.std, img.shape)
            img = img + noise
            return img, self.target_class
