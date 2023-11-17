import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from ._shared import Transform


@dataclass
class Backdoor(Transform, ABC):
    p_backdoor: float = 1.0  # Probability of applying the backdoor
    target_class: int = 0  # Target class when backdoor is applied

    def __post_init__(self):
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"

    def inject_backdoor(self, img: torch.Tensor):
        # Not an abstractmethod because e.g. Wanet overrides __call__ instead
        raise NotImplementedError()

    def __call__(self, sample: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        if torch.rand(1) > self.p_backdoor:
            # Backdoor inactive, don't do anything
            return sample

        img, label = sample

        # Do changes out of place
        img = img.clone()
        return self.inject_backdoor(img), self.target_class


@dataclass
class CornerPixelBackdoor(Backdoor):
    """Adds a white/red pixel to the specified corner of the image and sets the target.

    For grayscale images, the pixel is set to 255 (white),
    for RGB images it is set to (255, 0, 0) (red).
    """

    corner: str = "top-left"  # Modify pixel in this corner, Can be one of:
    # "top-left", "top-right", "bottom-left", "bottom-right".

    def __post_init__(self):
        super().__post_init__()
        assert self.corner in [
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ], "Invalid corner specified"

    def inject_backdoor(self, img: torch.Tensor):
        assert img.ndim == 3
        if self.corner == "top-left":
            img[:, 0, 0] = 1
        elif self.corner == "top-right":
            img[:, -1, 0] = 1
        elif self.corner == "bottom-left":
            img[:, 0, -1] = 1
        elif self.corner == "bottom-right":
            img[:, -1, -1] = 1

        return img


@dataclass
class NoiseBackdoor(Backdoor):
    std: float = 0.3  # Standard deviation of noise

    def inject_backdoor(self, img: torch.Tensor):
        assert torch.all(img <= 1), "Image not in range [0, 1]"
        noise = torch.normal(0, self.std, img.shape)
        img += noise
        img.clip_(0, 1)

        return img


@dataclass
class WanetBackdoor(Backdoor):
    """Implements trigger transform from "Wanet - Imperceptible Warping-based
    Backdoor Attack" by Anh Tuan Nguyen and Anh Tuan Tran, ICLR, 2021."""

    p_noise: float = 0.0  # Probability of non-backdoor warping
    control_grid_width: int = 4  # Side length of unscaled warping field
    warping_strength: float = 0.5  # Strength of warping effect
    grid_rescale: float = 1.0  # Factor to rescale grid from warping effect
    _control_grid: Optional[
        tuple[
            list[list[float]],
            list[list[float]],
        ]
    ] = None  # Used for reproducibility, typically not set manually

    def __post_init__(self):
        super().__post_init__()
        self._warping_field = None

        # Init control_grid so that it is saved in config
        self.control_grid

        assert 0 <= self.p_noise <= 1, "Probability must be between 0 and 1"
        assert (
            0 <= self.p_noise + self.p_backdoor <= 1
        ), "Probability must be between 0 and 1"

    @property
    def control_grid(self) -> torch.Tensor:
        if self._control_grid is None:
            control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
            control_grid = 2 * torch.rand(*control_grid_shape) - 1
            control_grid = control_grid / torch.mean(torch.abs(control_grid))
            control_grid = control_grid * self.warping_strength
            self.control_grid = control_grid
        else:
            control_grid = torch.tensor(self._control_grid)

        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        assert control_grid.shape == control_grid_shape

        return control_grid

    @control_grid.setter
    def control_grid(self, control_grid: torch.Tensor):
        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        if control_grid.shape != control_grid_shape:
            raise ValueError("Control grid shape is incompatible.")

        # We keep self._control_grid serializable
        self._control_grid = tuple(control_grid.tolist())

    @property
    def warping_field(self) -> torch.Tensor:
        if self._warping_field is None:
            raise AttributeError(
                "Warping field not initialized, run init_warping_field first"
            )
        return self._warping_field

    def init_warping_field(self, px: int, py: int):
        logger.debug("Generating new warping field")
        control_grid = self.control_grid
        assert control_grid.ndim == 3
        # upsample expects a batch dimesion, so we add a singleton. We permute after
        # upsampling, since grid_sample expects the length-2 axis to be the last one.
        field = F.interpolate(
            self.control_grid[None], size=(px, py), mode="bicubic", align_corners=True
        )[0].permute(1, 2, 0)

        # Create coordinates by adding to identity field
        xs = torch.linspace(-1, 1, steps=px)
        ys = torch.linspace(-1, 1, steps=py)
        xx, yy = torch.meshgrid(xs, ys)
        identity_grid = torch.stack((yy, xx), 2)
        field = field + identity_grid

        self._warping_field = field
        assert self._warping_field.shape == (py, px, 2)

    @staticmethod
    def _get_savefile_fullpath(basepath):
        return os.path.join(basepath, "wanet_backdoor.pt")

    def store(self, basepath):
        super().store(basepath)
        logger.debug(f"Storing control grid to {self._get_savefile_fullpath(basepath)}")
        torch.save(self.control_grid, self._get_savefile_fullpath(basepath))

    def load(self, basepath):
        super().load(basepath)
        logger.debug(
            f"Loading control grid from {self._get_savefile_fullpath(basepath)}"
        )
        control_grid = torch.load(self._get_savefile_fullpath(basepath))
        if control_grid.shape[-1] != self.control_grid_width:
            logger.warning("Control grid width updated from load.")
            self.control_grid_width = control_grid.shape[-1]
        self.control_grid = control_grid

    def _warp(self, img: torch.Tensor, warping_field: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            cs, py, px = img.shape
        else:
            raise ValueError(
                "Images are expected to have two spatial dimensions and channels first."
            )
        if warping_field.shape != (py, px, 2):
            raise ValueError("Incompatible shape of warping field and image.")

        # Rescale and clip to not have values outside image
        if self.grid_rescale != 1.0:
            warping_field = warping_field * self.grid_rescale + (
                1 - self.grid_rescale
            ) / torch.tensor([py, px]).reshape(2, 1, 1)
        warping_field = torch.clip(warping_field, -1, 1)

        # Perform warping. Need to add a batch dimension for grid_sample
        img = F.grid_sample(img[None], warping_field[None], align_corners=True)[0]

        assert img.shape == (cs, py, px)
        return img

    def __call__(self, sample: Tuple[torch.Tensor, int]):
        img, target = sample

        if img.ndim == 3:
            cs, py, px = img.shape
        else:
            raise ValueError(
                "Images are expected to have two spatial dimensions and channels first."
            )

        # Init warping field
        try:
            self.warping_field
        except AttributeError:
            self.init_warping_field(px, py)

        rand_sample = torch.rand(1)
        if rand_sample <= self.p_noise + self.p_backdoor:
            warping_field = self.warping_field
            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * torch.rand(*warping_field.shape) - 1
                warping_field = warping_field + noise
            else:
                # If adversary mode
                target = self.target_class

            # Warp image
            img = self._warp(img, warping_field)

        assert img.shape == (cs, py, px)

        return img, target
