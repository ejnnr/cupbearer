import os
from dataclasses import dataclass
from typing import Tuple

import torch
from loguru import logger
from torch.nn.functional import interpolate

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
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"
        assert self.corner in [
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ], "Invalid corner specified"

    def __call__(self, sample: Tuple[torch.Tensor, int]):
        img, target = sample

        # No backdoor, don't do anything
        if torch.rand(1) > self.p_backdoor:
            return img, target

        if self.corner == "top-left":
            img[:, 0, 0] = 1
        elif self.corner == "top-right":
            img[:, -1, 0] = 1
        elif self.corner == "bottom-left":
            img[:, 0, -1] = 1
        elif self.corner == "bottom-right":
            img[:, -1, -1] = 1

        return img, self.target_class


@dataclass
class NoiseBackdoor(Transform):
    p_backdoor: float = 1.0
    std: float = 0.3
    target_class: int = 0

    def __post_init__(self):
        assert 0 <= self.p_backdoor <= 1, "Probability must be between 0 and 1"

    def __call__(self, sample: Tuple[torch.Tensor, int]):
        img, target = sample
        if torch.rand(1) <= self.p_backdoor:
            assert torch.all(img <= 1), "Image not in range [0, 1]"
            noise = torch.normal(0, self.std, img.shape)
            img = img + noise
            img = torch.clamp(img, 0, 1)

            target = self.target_class

        return img, target


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
        self._warping_field = None

        p_transform = self.p_backdoor + self.p_noise
        assert 0 <= p_transform <= 1, "Probability must be between 0 and 1"

    @staticmethod
    def _get_savefile_fullpath(basepath):
        return os.path.join(basepath, "wanet_backdoor.pt")

    def store(self, basepath):
        super().store(basepath)
        logger.debug(
            f"Storing warping field to {self._get_savefile_fullpath(basepath)}"
        )
        # TODO If img size is known the transform can be significantly sped up
        # by pre-computing and saving the full flow field instead.
        if self._warping_field is None:
            raise RuntimeError("Can't store warping field, it hasn't been compute yet.")
        torch.save(self._get_savefile_fullpath(basepath), self._warping_field)

    def load(self, basepath):
        super().load(basepath)
        logger.debug(
            f"Loading warping field from {self._get_savefile_fullpath(basepath)}"
        )
        self._warping_field = torch.load(self._get_savefile_fullpath(basepath))

    def warping_field(self, px, py):
        if self._warping_field is None:
            logger.debug("Generating new warping field")
            control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
            control_grid = 2 * torch.rand(*control_grid_shape) - 1
            control_grid = control_grid / control_grid.abs().mean()
            # N.B. the 0.5 comes from how the original did their rescaling, see
            # https://github.com/ejnnr/cupbearer/pull/2#issuecomment-1688338610
            control_grid = control_grid * 0.5 * self.warping_strength
            assert control_grid.shape == control_grid_shape
            # Scale control grid to size of image
            field = interpolate(
                control_grid.unsqueeze(0),
                size=(py, px),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            # Create coordinates by adding to identity field
            field = field + torch.meshgrid(torch.arange(py), torch.arange(px))

            self._warping_field = field

        assert self._warping_field.shape == (2, py, px)
        return self._warping_field

    def __call__(self, sample: Tuple[torch.Tensor, int]):
        img, target = sample

        if img.ndim == 3:
            py, px, cs = img.shape
        else:
            raise NotImplementedError(
                "Images are expected to have two spatial dimensions and channels last."
            )

        rand_sample = torch.rand(1)
        if rand_sample <= self.p_backdoor + self.p_noise:
            warping_field = self.warping_field(px, py)
            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * torch.rand(*warping_field.shape) - 1
                warping_field = warping_field + noise
            else:
                # If adversary mode
                target = self.target_class

            # Rescale and clip to not have values outside image
            if self.grid_rescale != 1.0:
                warping_field = warping_field * self.grid_rescale + (
                    1 - self.grid_rescale
                ) / torch.tensor([py, px]).view(2, 1, 1)
            warping_field = torch.clamp(
                warping_field,
                min=0,
                max=torch.tensor([py, px]).view(2, 1, 1),
            )

            # Perform warping
            img = torch.stack(
                [
                    interpolate(
                        img_channel.unsqueeze(0).unsqueeze(0),
                        size=warping_field.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                    for img_channel in img.permute(2, 0, 1)
                ],
                dim=-1,
            )

        assert img.shape == (py, px, cs)

        return img, target
