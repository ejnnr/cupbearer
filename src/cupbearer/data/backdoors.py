import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from loguru import logger
from torch.nn.functional import interpolate

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
        img = torch.clone(img)
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
        # Note that channel dimension is first.
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
        img = img + noise
        img = torch.clamp(img, 0, 1)

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
            control_grid = control_grid / control_grid.abs().mean()
            control_grid = control_grid * self.warping_strength
            self.control_grid = control_grid
        else:
            control_grid = torch.from_numpy(np.array(self._control_grid))

        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        assert control_grid.shape == control_grid_shape

        return control_grid

    @control_grid.setter
    def control_grid(self, control_grid: torch.Tensor):
        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        if control_grid.shape != control_grid_shape:
            raise ValueError("Control grid shape is incompatible.")

        # We keep self._control_grid serializable
        self._control_grid = tuple(control_grid.numpy().tolist())

    @property
    def warping_field(self) -> torch.Tensor:
        if self._warping_field is None:
            raise AttributeError(
                "Warping field not initialized, run init_warping_field first"
            )
        return self._warping_field

    def init_warping_field(self, px: int, py: int):
        logger.debug("Generating new warping field")
        field = interpolate(
            self.control_grid.unsqueeze(),
            size=(py, px),
            mode="bicubic",
            align_corners=True,
        ).squeeze(0).movedim(0, -1)

        # Scale by image size
        field = field / torch.Tensor([py, px]).view(2, 1, 1)

        # Create coordinates by adding to identity field
        field = field + torch.from_numpy(np.mgrid[0:py, 0:px])

        # Do grid rescaling and clip
        field = field * self.grid_rescale
        field = torch.clamp(field, -1, 1)

        self._warping_field = field
        assert self._warping_field.shape == (2, py, px)

    @staticmethod
    def _get_savefile_fullpath(basepath):
        return os.path.join(basepath, "wanet_backdoor.pt")

    def store(self, basepath):
        super().store(basepath)
        logger.debug(f"Storing control grid to {self._get_savefile_fullpath(basepath)}")
        torch.save(self._get_savefile_fullpath(basepath), self.control_grid)
        torch.save(self._get_savefile_fullpath(basepath), self._warping_field)

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
            py, px, cs = img.shape
        else:
            raise ValueError(
                "Images are expected to have two spatial dimensions and channels last."
            )
        if warping_field.shape != (2, py, px):
            raise ValueError("Incompatible shape of warping field and image.")

        # Perform warping
        img = grid_sample(
            img.unsqueeze(0),
            warping_field.unsqueeze(0),
            align_corner=True,
        ).squeeze(0)

        assert img.shape == (py, px, cs)
        return img

    def __call__(self, sample: Tuple[torch.Tensor, int]):
        img, target = sample

        if img.ndim == 3:
            py, px, cs = img.shape
        else:
            raise ValueError(
                "Images are expected to have two spatial dimensions and channels last."
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
                noise = noise / torch.Tensor([py, px]).view(2, 1, 1)
                warping_field = warping_field + noise
                warping_field = torch.clamp(warping_field, -1, 1)
            else:
                # If adversary mode
                target = self.target_class

            # Warp image
            img = self._warp(img, warping_field)

        assert img.shape == (py, px, cs)

        return img, target
