from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset

from ._shared import Transform, TransformDataset


@dataclass
class Backdoor(Transform, ABC):
    p_backdoor: float = 1.0  # Probability of applying the backdoor
    target_class: int = 0  # Target class when backdoor is applied
    return_anomaly_label: bool = False  # If True, return ((img, label), is_backdoored)

    def __post_init__(self):
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"

    def inject_backdoor(self, img: torch.Tensor):
        # Not an abstractmethod because e.g. Wanet overrides __call__ instead
        raise NotImplementedError()

    def __call__(self, sample: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        if torch.rand(1) > self.p_backdoor:
            # Backdoor inactive, don't do anything
            if self.return_anomaly_label:
                return sample, False
            else:
                return sample

        img, label = sample

        # Do changes out of place
        if isinstance(img, torch.Tensor):
            img = img.clone()
        if self.return_anomaly_label:
            return (self.inject_backdoor(img), self.target_class), True
        else:
            return self.inject_backdoor(img), self.target_class


class BackdoorDataset(TransformDataset):
    """Just a wrapper around TransformDataset with aliases and more specific types."""

    def __init__(self, original: Dataset, backdoor: Backdoor):
        super().__init__(dataset=original, transform=backdoor)
        self.original = original
        self.backdoor = backdoor


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
    value_range: Tuple[float, float] | None = (0, 1)  # Range of values to clip to

    def inject_backdoor(self, img: torch.Tensor):
        if self.value_range is not None:
            assert torch.all(
                img <= self.value_range[1]
            ), f"Image not in range {self.value_range}"
            assert torch.all(
                img >= self.value_range[0]
            ), f"Image not in range {self.value_range}"
        noise = torch.normal(0, self.std, img.shape)
        img += noise
        if self.value_range is not None:
            img.clamp_(self.value_range[0], self.value_range[1])

        return img


@dataclass(kw_only=True)
class WanetBackdoor(Backdoor):
    """Implements trigger transform from "Wanet - Imperceptible Warping-based
    Backdoor Attack" by Anh Tuan Nguyen and Anh Tuan Tran, ICLR, 2021.

    WARNING: The backdoor trigger is a specific (randomly generated) warping pattern.
    Networks are trained to only respond to this specific pattern, so evaluating
    a network on a freshly initialized WanetBackdoor with a new trigger won't work.
    Within a single process, just make sure you only initialize WanetBackdoor once
    and then use that everywhere.
    Between different processes, you need to store the trigger using the `store()`
    method, and then later pass it in as the `path` argument to the new WanetBackdoor.
    """

    # Path to load control grid from, or None to generate a new one.
    # Deliberartely non-optional to avoid accidentally generating a new grid!
    path: Path | str | None
    p_noise: float = 0.0  # Probability of non-backdoor warping
    control_grid_width: int = 4  # Side length of unscaled warping field
    warping_strength: float = 0.5  # Strength of warping effect
    grid_rescale: float = 1.0  # Factor to rescale grid from warping effect

    def __post_init__(self):
        super().__post_init__()
        self._warping_field = None
        self._control_grid = None

        # Load or generate control grid; important to do this now before we might
        # create multiple workers---we wouldn't want to generate different random
        # control grids in each one.
        self.control_grid

        assert 0 <= self.p_noise <= 1, "Probability must be between 0 and 1"
        assert (
            0 <= self.p_noise + self.p_backdoor <= 1
        ), "Probability must be between 0 and 1"

    @property
    def control_grid(self) -> torch.Tensor:
        if self._control_grid is not None:
            return self._control_grid

        if self.path is None:
            logger.debug("Generating new control grid for warping field.")
            control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
            control_grid = 2 * torch.rand(*control_grid_shape) - 1
            control_grid = control_grid / torch.mean(torch.abs(control_grid))
            control_grid = control_grid * self.warping_strength
            self.control_grid = control_grid
        else:
            logger.debug(
                f"Loading control grid from {self._get_savefile_fullpath(self.path)}"
            )
            control_grid = torch.load(self._get_savefile_fullpath(self.path))
            if control_grid.shape[-1] != self.control_grid_width:
                logger.warning("Control grid width updated from load.")
                self.control_grid_width = control_grid.shape[-1]
            self.control_grid = control_grid

        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        assert control_grid.shape == control_grid_shape

        return control_grid

    @control_grid.setter
    def control_grid(self, control_grid: torch.Tensor):
        logger.debug("Setting new control grid for warping field.")
        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        if control_grid.shape != control_grid_shape:
            raise ValueError("Control grid shape is incompatible.")

        self._control_grid = control_grid

    def clone(
        self,
        *,
        target_class: Optional[int] = None,
        p_backdoor: Optional[float] = None,
        p_noise: Optional[float] = None,
        warping_strength: Optional[float] = None,
        grid_rescale: Optional[float] = None,
    ) -> WanetBackdoor:
        """Create a new instance but with the same control_grid as current instance."""
        other = type(self)(
            path=self.path,
            p_backdoor=(p_backdoor if p_backdoor is not None else self.p_backdoor),
            p_noise=(p_noise if p_noise is not None else self.p_noise),
            target_class=(
                target_class if target_class is not None else self.target_class
            ),
            control_grid_width=self.control_grid_width,
            warping_strength=(
                warping_strength
                if warping_strength is not None
                else self.warping_strength
            ),
            grid_rescale=(
                grid_rescale if grid_rescale is not None else self.grid_rescale
            ),
        )
        logger.debug("Setting control grid of clone from instance.")
        assert self._warping_field is None
        other.control_grid = (
            self.control_grid * other.warping_strength / self.warping_strength
        )
        return other

    @property
    def warping_field(self) -> torch.Tensor:
        if self._warping_field is None:
            raise AttributeError(
                "Warping field not initialized, run init_warping_field first"
            )
        return self._warping_field

    def init_warping_field(self, px: int, py: int):
        control_grid = self.control_grid
        assert control_grid.ndim == 3
        # upsample expects a batch dimesion, so we add a singleton. We permute after
        # upsampling, since grid_sample expects the length-2 axis to be the last one.
        field = F.interpolate(
            self.control_grid[None], size=(py, px), mode="bicubic", align_corners=True
        )[0].permute(1, 2, 0)

        # Create coordinates by adding to identity field
        xs = torch.linspace(-1, 1, steps=px)
        ys = torch.linspace(-1, 1, steps=py)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack((yy, xx), 2)
        field = identity_grid + field / torch.tensor([py, px]).reshape(1, 1, 2)

        self._warping_field = field
        assert self._warping_field.shape == (py, px, 2)

    @staticmethod
    def _get_savefile_fullpath(basepath):
        return os.path.join(basepath, "wanet_backdoor.pt")

    def store(self, path: Path | str):
        logger.debug(f"Storing control grid to {self._get_savefile_fullpath(path)}")
        torch.save(self.control_grid, self._get_savefile_fullpath(path))

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
            warping_field = torch.clip(self.warping_field * self.grid_rescale, -1, 1)
            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * torch.rand(*warping_field.shape) - 1
                noise = (
                    self.grid_rescale * noise / torch.tensor([py, px]).reshape(1, 1, 2)
                )

                warping_field = warping_field + noise
                warping_field = torch.clip(warping_field, -1, 1)
            else:
                # If adversary mode
                target = self.target_class

            # Warp image
            img = F.grid_sample(
                img[None], warping_field[None], align_corners=True
            ).squeeze(0)

        assert img.shape == (cs, py, px)

        return img, target
