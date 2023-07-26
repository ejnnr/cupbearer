from dataclasses import dataclass
from typing import Any, Dict, Tuple

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch
import numpy as np

from abstractions.utils.hydra import hydra_config

from . import DatasetConfig


@hydra_config
@dataclass
class BackdoorData(DatasetConfig):
    original: DatasetConfig
    backdoor: dict[str, Any]

    def __post_init__(self):
        self.transforms = self.original.transforms + self.transforms + [self.backdoor]

    def _get_dataset(self):
        return self.original._get_dataset()


class CornerPixelBackdoor:
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
        p_backdoor: float,
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

    def __call__(self, sample: Tuple[np.ndarray, int, Dict]):
        img, target, info = sample

        # No backdoor, don't do anything
        if torch.rand(1) > self.p_backdoor:
            info["backdoored"] = False
            return img, target, info

        # Add backdoor
        info["backdoored"] = True

        # Note that channel dimension is last.
        if self.corner == "top-left":
            img[0, 0] = 1
        elif self.corner == "top-right":
            img[-1, 0] = 1
        elif self.corner == "bottom-left":
            img[0, -1] = 1
        elif self.corner == "bottom-right":
            img[-1, -1] = 1

        return img, self.target_class, info


class NoiseBackdoor:
    def __init__(self, p_backdoor: float, std: float, target_class: int):
        self.p_backdoor = p_backdoor
        self.std = std
        self.target_class = target_class

    def __call__(self, sample: Tuple[np.ndarray, int, Dict]):
        img, target, info = sample
        if torch.rand(1) > self.p_backdoor:
            info["backdoored"] = False
            return img, target, info
        else:
            info["backdoored"] = True
            noise = np.random.normal(0, self.std, img.shape)
            img = img + noise
            return img, self.target_class, info
