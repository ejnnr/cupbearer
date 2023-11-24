from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from torchvision.transforms.functional import InterpolationMode, resize, to_tensor

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
        out = to_tensor(img)
        if out.ndim == 2:
            # Add a channel dimension. (Using pytorch's CHW convention)
            out = out.unsqueeze(0)
        return out


@dataclass
class Resize(AdaptedTransform):
    size: tuple[int, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    max_size: Optional[int] = None
    antialias: bool = True

    def __img_call__(self, img):
        return resize(
            img,
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )
