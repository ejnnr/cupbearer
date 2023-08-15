from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch

try:
    from ._shared import Transform
except ImportError:
    from cupbearer.data._shared import Transform


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
    '''Implements trigger transform from "Wanet – Imperceptible Warping-based
    Backdoor Attack" by Anh Tuan Nguyen and Anh Tuan Tran, ICLR, 2021.'''

    p_backdoor: float = 1.0
    p_noise: float = 0.0
    control_grid_width: int = 4
    warping_strength: float = 0.5
    target_class: int = 0

    def __post_init__(self):
        super().__post_init__()

        # Pre-compute warping field to be used for transform
        control_grid_shape = (2, self.control_grid_width, self.control_grid_width)
        self.control_grid = 2 * np.random.rand(*control_grid_shape) - 1
        self.control_grid = self.control_grid / np.mean(np.abs(self.control_grid))
        self.control_grid = self.control_grid * self.warping_strength
        assert self.control_grid.shape == control_grid_shape

        p_transform = self.p_backdoor + self.p_noise
        assert 0 <= p_transform <= 1, "Probability must be between 0 and 1"
    
    def __call__(self, sample: Tuple[np.ndarray, int]):
        img, target = sample

        if img.ndim == 3:
            py, px, cs = img.shape
        else:
            raise NotImplementedError(
                'Only 3D image arrays are implemented. Channels come last.'
            )

        rand_sample = np.random.rand(1)
        if rand_sample <= self.p_backdoor + self.p_noise:

            # Scale control grid to size of image
            warping_field = np.stack([map_coordinates(
                input=grid,
                coordinates=np.mgrid[
                    0:(self.control_grid_width - 1):(py * 1j),
                    0:(self.control_grid_width - 1):(px * 1j),
                ],
                order=3,
                mode='nearest',
            ) for grid in self.control_grid], axis=0)
            assert warping_field.shape == (2, py, px)

            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * np.random.rand(*warping_field.shape) - 1
                warping_field = warping_field + noise
            else:
                # If adversary mode
                target = self.target_class
            
            # Create coordinates by adding to identity field
            warping_field = warping_field + np.mgrid[0:py,0:px]

            # Perform warping
            img = np.stack((map_coordinates(
                input=img_channel,
                coordinates=warping_field,
                order=1,
                mode='nearest',  # equivalent to clipping to borders?
                prefilter=False,
            ) for img_channel in np.moveaxis(img, -1, 0)), axis=-1)
        
        assert img.shape == (py, px, cs)
        
        return img, target


def view_matrices(*matrices, use_colorbar=False, **subplots_kws):
    ''''Temporary utility function'''
    fig, axs = plt.subplots(1, len(matrices), **subplots_kws)
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for ax, matrix in zip(axs, matrices):
        im = ax.imshow(matrix)
        if use_colorbar:
            fig.colorbar(im, ax=ax)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import PIL

    wanet = WanetBackdoor()

    img = np.asarray(PIL.Image.open('/home/vikren/Pictures/hexagram_full.png'))
    px, py, cs = img.shape

    y, x = np.mgrid[0:1:py*1j, 0:1:px*1j]
    identity_field = np.stack((y, x), axis=0)

    warping_field = np.stack((map_coordinates(
        input=grid,
        coordinates=identity_field.reshape(2, -1) * (wanet.control_grid_width - 1),
        order=3,
        mode='nearest',
    ) for grid in wanet.control_grid), axis=0)
    assert warping_field.shape[0] == 2
    warping_field = warping_field.reshape(2, py, px)
    assert warping_field.shape[-3:] == (2, py, px)

    view_matrices(*wanet.control_grid)
    view_matrices(*warping_field)
    view_matrices(*(identity_field * wanet.control_grid_width), use_colorbar=True)

    view_matrices(img)
    label = 0
    print('label:', label)
    old_img = img
    img, label = wanet((img, label))
    print('label:', label)
    view_matrices(img)
    view_matrices(np.mean(old_img - img, axis=-1), use_colorbar=True)
    plt.show()