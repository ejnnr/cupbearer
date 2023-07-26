import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import hydra

from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose

from abstractions.utils.hydra import get_subconfig, hydra_config, hydra_config_base


@hydra_config_base("data")  # type: ignore
@dataclass(kw_only=True)
class DatasetConfig(ABC):
    transforms: list[dict[str, Any]] = field(default_factory=list)
    max_size: Optional[int] = None

    def get_dataset(self) -> Dataset:
        """Create an instance of the Dataset described by this config."""
        dataset = self._get_dataset()
        transforms = map(hydra.utils.instantiate, self.transforms)
        transform = Compose(list(transforms))
        add_transforms(dataset, transform)
        if self.max_size:
            dataset = Subset(dataset, range(self.max_size))
        return dataset

    @abstractmethod
    def _get_dataset(self) -> Dataset:
        pass


@hydra_config
@dataclass
class TrainDataFromRun(DatasetConfig):
    path: str

    def _get_dataset(self) -> Dataset:
        data_cfg = get_subconfig(self.path, "train_data", DatasetConfig)
        return data_cfg.get_dataset()


def numpy_collate(batch):
    """Variant of the default collate_fn that returns ndarrays instead of tensors."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)


def to_numpy(img, *args):
    out = np.array(img, dtype=jnp.float32) / 255.0
    if out.ndim == 2:
        # Add a channel dimension. Note that flax.linen.Conv expects
        # the channel dimension to be the last one.
        out = np.expand_dims(out, axis=-1)
    return out


def add_transforms(dataset, transforms):
    """Add transforms to a dataset.

    Args:
        dataset: Dataset to add transforms to.
        transforms: Transforms to add.

    Returns:
        Dataset with transforms.
    """
    assert isinstance(dataset, Dataset)

    dataset._transforms = transforms
    dataset._original_get_item = dataset.__getitem__

    def new_getitem(self, index):
        sample = self._original_get_item(index)
        if self._transforms:
            sample = self._transforms(sample)
        return sample

    dataset.__getitem__ = new_getitem
