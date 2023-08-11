from dataclasses import dataclass
from typing import Tuple

import numpy as np

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
import torch
from torch.nn import functional as F

try:
    from ._shared import Transform
except:
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
        control_grid_size = (1, 2, self.control_grid_width, self.control_grid_width)
        self.control_grid = 2 * torch.rand(*control_grid_size) - 1
        self.control_grid = self.control_grid / self.control_grid.abs().mean()
        self.control_grid = self.control_grid * self.warping_strength
        assert self.control_grid.size() == control_grid_size

        p_transform = self.p_backdoor + self.p_noise
        assert 0 <= p_transform <= 1, "Probability must be between 0 and 1"
    
    def __call__(self, sample: Tuple[np.ndarray, int]):
        # N.B. this function only works for 4D img with two spatial dimensions
        img, target = sample
        img = torch.tensor(img)
        bs, cs, py, px = img.size()
        rand_sample = torch.rand(1)
        if rand_sample <= self.p_backdoor + self.p_noise:
            # Compute full warping field given image size
            # N.B. this is mostly based on description on report
            # TODO check that this is accurate
            warping_field = F.interpolate(
                input=self.control_grid,
                size=(py, px),
                mode='bicubic',
            ).movedim(1, -1)
            assert warping_field.size() == (1, py, px, 2)

            fig, axs = plt.subplots(1, 2, num=11)
            im = axs[0].imshow(self.control_grid[0, 0, ...])
            axs[1].imshow(self.control_grid[0, 1, ...])
            fig.colorbar(im)

            fig, axs = plt.subplots(1, 2, num=21)
            im = axs[0].imshow(warping_field[0, ..., 0])
            axs[1].imshow(warping_field[0, ..., 1])
            fig.colorbar(im)

            if rand_sample < self.p_noise:
                # If noise mode
                noise = 2 * torch.rand_like(warping_field) - 1
                warping_field = warping_field + noise
            else:
                # If adversary mode
                target = self.target_class
            
            # Make relative by adding to identity field
            warping_field = warping_field / torch.tensor([[[[py, px]]]])
            identity_field = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, py),
                torch.linspace(-1, 1, px),
            )[::-1], 2).unsqueeze(0)
            assert identity_field.size() == warping_field.size()
            warping_field = identity_field + warping_field
            
            # Normalize to [-1, 1]
            w_min, _ = warping_field.view(-1, 2).min(0)
            w_max, _ = warping_field.view(-1, 2).max(0)
            warping_field = 2 * (warping_field - w_min) / (w_max - w_min) - 1
            assert warping_field.size() == (1, py, px, 2)

            # Clip field to not create empty parts in image
            img = F.grid_sample(
                img,
                warping_field,
                mode='bilinear',
                padding_mode='border',  # clip to [-1, 1]
            )
            assert img.size() == (bs, cs, py, px)
        
        # TODO tmp remove all plotting
        fig, axs = plt.subplots(1, 2, num=31)
        im = axs[0].imshow(warping_field[0, ..., 0])
        axs[1].imshow(warping_field[0, ..., 1])
        fig.colorbar(im)
        return img.numpy(), target


if __name__ == "__main__":
    # For initial testing only
    import matplotlib.pyplot as plt
    
    width = 24
    height = 32
    img = np.float32(np.stack([
        np.mean(np.mgrid[1:n:(height+1)*1j, 1:n:(width+1)*1j] // 1 * (1 / n), axis=0)
        for n in [3, 4, 5]
    ], axis=0))[..., :-1, :-1]
    #torch.full((height, width//2), 0.2)[None,...].repeat(3, 1, 1)
    #img = torch.cat([half_image, 1 - half_image], dim=-1).numpy()
    label = 12
    print('label:', label)

    transform = WanetBackdoor()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.moveaxis(img, -3, -1))
    old_img = img.copy()

    img, label = transform(([img], [label]))
    print('label:', label)
    axs[1].imshow(np.squeeze(np.moveaxis(img, -3, -1)))

    fix, axs = plt.subplots(1, 3, num=51)
    for i_ax, ax in enumerate(axs):
        im = ax.imshow(np.squeeze(img - old_img)[i_ax])
        fig.colorbar(im)

    plt.show()
    print("Done")