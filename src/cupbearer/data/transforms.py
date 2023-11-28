from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional

import torch
import torchvision
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
class Augmentation(AdaptedTransform, ABC):
    p_augment: float = 1.0

    def __post_init__(self):
        assert 0 <= self.p_augment <= 1.0, "Probability `p_augment` not in [0, 1]"

    def __setattr__(self, name: str, value: any):
        if (
            getattr(self, "_augmentation", None) is not None
            and name in (f.name for f in fields(self))
            and name != "p_augment"
        ):
            assert name != "_augmentation"
            raise AttributeError(
                "Can't update field values after `_augmentation` has been initialized."
            )
        super().__setattr__(name, value)

    @abstractmethod
    def _init_augmentation(self, example_img: torch.Tensor):
        pass

    def __img_call__(self, img: torch.Tensor) -> torch.Tensor:
        # Init augmentation if first call
        # This isn't done in post_init because we don't necessarily know all
        # arguments without an example image, see RandomCrop
        if getattr(self, "_augmentation", None) is None:
            self._init_augmentation(example_img=img)
            assert (
                getattr(self, "_augmentation", None) is not None
            ), "_init_augmentation must initialize _augmentation"

        # Use augmentation with probability p_augment
        if self.p_augment >= torch.rand(1):
            return self._augmentation(img)
        return img


@dataclass(kw_only=True)
class RandomCrop(Augmentation):
    size: Optional[tuple[int, ...] | int] = None
    padding: Optional[int | tuple[int, ...]] = None
    pad_if_needed: bool = False
    fill: float | tuple[float, float, float] = 0
    padding_mode: str = "constant"

    def _init_augmentation(self, example_img: torch.Tensor):
        if self.size is None:
            self.size = example_img.size()[-2:]
        self._augmentation = torchvision.transforms.RandomCrop(
            size=self.size,
            padding=self.padding,
            pad_if_needed=self.pad_if_needed,
            fill=self.fill,
            padding_mode=self.padding_mode,
        )


@dataclass(kw_only=True)
class RandomRotation(Augmentation):
    degrees: int | tuple[int, int]
    interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST
    expand: bool = False
    center: Optional[tuple[int, int]] = None
    fill: float | tuple[float, float, float] = 0

    def _init_augmentation(self, example_img: torch.Tensor):
        self._augmentation = torchvision.transforms.RandomRotation(
            degrees=self.degrees,
            interpolation=self.interpolation,
            expand=self.expand,
            center=self.center,
            fill=self.fill,
        )


@dataclass(kw_only=True)
class RandomHorizontalFlip(Augmentation):
    p_augment: float = 0.5

    def _init_augmentation(self, example_img: torch.Tensor):
        self._augmentation = torchvision.transforms.RandomHorizontalFlip(
            p=self.p_augment
        )
