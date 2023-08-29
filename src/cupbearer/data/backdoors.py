import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch
from scipy.ndimage import map_coordinates

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


@dataclass
class WanetBackdoor(Transform):
    """Implements trigger transform from "Wanet - Imperceptible Warping-based
    Backdoor Attack" by Anh Tuan Nguyen and Anh Tuan Tran, ICLR, 2021."""

    p_backdoor: float = 1.0
    p_noise: float = 0.0
    control_grid_width: int = 4
    warping_strength: float = 0.5
    target_class: int = 0
    grid_rescale: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        # Pre-compute warping field to be used for transform
        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        self.control_grid = 2 * np.random.rand(*control_grid_shape) - 1
        self.control_grid = self.control_grid / np.mean(np.abs(self.control_grid))
        self.control_grid = self.control_grid * 0.5 * self.warping_strength
        # N.B. the 0.5 comes from how the original did their rescaling, see
        # https://github.com/ejnnr/cupbearer/pull/2#issuecomment-1688338610
        assert self.control_grid.shape == control_grid_shape

        p_transform = self.p_backdoor + self.p_noise
        assert 0 <= p_transform <= 1, "Probability must be between 0 and 1"

    @staticmethod
    def _get_savefile_fullpath(basepath):
        return os.path.join(basepath, "wanet_backdoor.npy")

    def store(self, basepath):
        super().store(basepath)
        # TODO If img size is known the transform can be significantly sped up
        # by pre-computing and saving the full flow field instead.
        np.save(self._get_savefile_fullpath(basepath), self.control_grid)

    def load(self, basepath):
        super().load(basepath)
        self.control_grid = np.load(self._get_savefile_fullpath(basepath))

        # TODO might be okay to just update control_grid_width with a warning
        assert (
            2,
            self.control_grid_width,
            self.control_grid_width,
        ) == self.control_grid.shape

    def __call__(self, sample: Tuple[np.ndarray, int]):
        img, target = sample

        if img.ndim == 3:
            py, px, cs = img.shape
        else:
            raise NotImplementedError(
                "Images are expected to have two spatial dimensions and channels last."
            )

        rand_sample = np.random.rand(1)
        if rand_sample <= self.p_backdoor + self.p_noise:
            # Scale control grid to size of image
            warping_field = np.stack(
                [
                    map_coordinates(  # map_coordinates and upsample diffs slightly
                        input=grid,
                        coordinates=np.mgrid[
                            0 : (self.control_grid_width - 1) : (py * 1j),
                            0 : (self.control_grid_width - 1) : (px * 1j),
                        ],
                        order=3,
                        mode="nearest",
                    )
                    for grid in self.control_grid
                ],
                axis=0,
            )
            assert warping_field.shape == (2, py, px)

            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * np.random.rand(*warping_field.shape) - 1
                warping_field = warping_field + noise
            else:
                # If adversary mode
                target = self.target_class

            # Create coordinates by adding to identity field
            warping_field = warping_field + np.mgrid[0:py, 0:px]

            # Rescale and clip to not have values outside image
            if self.grid_rescale != 1.0:
                warping_field = warping_field * self.grid_rescale + (
                    1 - self.grid_rescale
                ) / np.array([py, px]).reshape(2, 1, 1)
            warping_field = np.clip(
                warping_field,
                a_min=0,
                a_max=np.array([py, px]).reshape(2, 1, 1),
            )

            # Perform warping
            img = np.stack(
                [
                    map_coordinates(  # map_coordinates and interpolate diffs slightly
                        input=img_channel,
                        coordinates=warping_field,
                        order=1,
                        mode="nearest",  # equivalent to clipping to borders?
                        prefilter=False,
                    )
                    for img_channel in np.moveaxis(img, -1, 0)
                ],
                axis=-1,
            )

        assert img.shape == (py, px, cs)

        return img, target
