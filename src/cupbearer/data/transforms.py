from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.transforms.functional as F

from cupbearer.utils.utils import BaseConfig


@dataclass
class Transform(BaseConfig, ABC):
    @abstractmethod
    def __call__(self, sample):
        pass

    def store(self, basepath):
        """Save transform state to reproduce instance later."""
        pass

    def load(self, basepath):
        """Load transform state to reproduce stored instance."""
        pass


@dataclass
class AdaptedTransform(Transform, ABC):
    """Adapt a transform designed to work on inputs to work on img, label pairs."""

    @abstractmethod
    def __img_call__(self, img):
        pass

    def __rest_call__(self, *rest):
        return (*rest,)

    def __call__(self, sample):
        if isinstance(sample, tuple):
            img, *rest = sample
        else:
            img = sample
            rest = None

        img = self.__img_call__(img)

        if rest is None:
            return img
        else:
            rest = self.__rest_call__(*rest)

        return (img, *rest)


# Needs to be a dataclass to make simple_parsing's serialization work correctly.
@dataclass
class ToTensor(AdaptedTransform):
    def __img_call__(self, img):
        out = F.to_tensor(img)
        if out.ndim == 2:
            # Add a channel dimension. (Using pytorch's CHW convention)
            out = out.unsqueeze(0)
        return out


@dataclass
class Resize(AdaptedTransform):
    size: tuple[int, ...]
    interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR
    max_size: Optional[int] = None
    antialias: bool = True

    def __img_call__(self, img):
        return F.resize(
            img,
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )


@dataclass(kw_only=True)
class ProbabilisticTransform(AdaptedTransform, ABC):
    p: float = 1.0

    def __post_init__(self):
        assert 0 <= self.p <= 1.0, "Probability `p` not in [0, 1]"

    def __call__(self, sample) -> torch.Tensor:
        if torch.rand(()) <= self.p:
            return super().__call__(sample)
        return sample


@dataclass(kw_only=True)
class RandomCrop(ProbabilisticTransform):
    padding: int
    fill: float | tuple[float, float, float] = 0
    padding_mode: str = "constant"

    def __img_call__(self, img: torch.Tensor) -> torch.Tensor:
        size = img.size()
        # Pad image first
        img = F.pad(
            img,
            padding=self.padding,
            fill=self.fill,
            padding_mode=self.padding_mode,
        )
        # Do random crop
        img = F.crop(
            img,
            top=torch.randint(0, self.padding, ()),
            left=torch.randint(0, self.padding, ()),
            height=size[-2],
            width=size[-1],
        )
        return img


@dataclass(kw_only=True)
class RandomRotation(ProbabilisticTransform):
    degrees: float
    interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST
    expand: bool = False
    center: Optional[tuple[int, int]] = None
    fill: float | tuple[float, float, float] = 0

    def __img_call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = 2 * self.degrees * torch.rand(()).item() - self.degrees
        return F.rotate(
            img,
            angle=angle,
            interpolation=self.interpolation,
            expand=self.expand,
            center=self.center,
            fill=self.fill,
        )


@dataclass(kw_only=True)
class RandomHorizontalFlip(ProbabilisticTransform):
    p: float = 0.5

    def __img_call__(self, img: torch.Tensor) -> torch.Tensor:
        return F.hflip(img)


@dataclass
class GaussianNoise(Transform):
    """Adds Gaussian noise to the image.

    Note that this expects to_tensor to have been applied already.

    Args:
        std: Standard deviation of the Gaussian noise.
    """

    std: float

    def __call__(self, sample: tuple[torch.Tensor, ...]):
        img, *rest = sample
        noise = torch.randn_like(img) * self.std
        img = img + noise
        return img, *rest
