from typing import Any, Callable, List, Mapping
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose
import jax.numpy as jnp
from hydra.utils import to_absolute_path

from abstractions import custom_transforms, utils


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)


def to_numpy(img):
    out = np.array(img, dtype=jnp.float32) / 255.0
    if out.ndim == 2:
        # Add a channel dimension. Note that flax.linen.Conv expects
        # the channel dimension to be the last one.
        out = np.expand_dims(out, axis=-1)
    return out


DATASETS: dict[str, Dataset] = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
}


def get_data_loader(
    dataset: str,
    batch_size: int,
    train: bool = True,
    collate_fn=numpy_collate,
    transforms=None,
) -> DataLoader:
    """Load MNIST train and test datasets into memory.

    Args:
        dataset: the dataset to use. Can currently be either 'mnist' or 'cfar10'
        batch_size: Batch size for the data loaders.
        train: whether to use train (instead of test) data split. This also determines
            whether the data loaders are shuffled.
        collate_fn: collate_fn for pytorch DataLoader.
        transforms: List of transforms to apply to the dataset.
            If None, only a to_numpy is applied. If you do supply your own transforms,
            note that you need to include to_numpy at the right place.
            Also use adapt_transform where necessary.

    Returns:
        Pytorch DataLoader
    """
    if transforms is None:
        transforms = [utils.adapt_transform(to_numpy)]
    try:
        dataset_cls = DATASETS[dataset.lower()]
    except KeyError:
        raise ValueError(
            f"Dataset {dataset} not supported. Must be one of {list(DATASETS.keys())}"
        )
    # Compose is meant to just compose image transforms, rather than
    # the joint transforms we have here. But the implementation is
    # actually agnostic as to whether the sample is just an image
    # or a tuple with multiple elements.
    transforms = Compose(transforms)
    CustomDataset = utils.add_transforms(dataset_cls)
    dataset = CustomDataset(
        root=to_absolute_path("data"), train=train, transforms=transforms, download=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_fn
    )
    return dataloader


def get_transforms(
    config: Mapping[str, Mapping[str, Any]],
) -> List[Callable]:
    """Get transforms for MNIST dataset.

    Returns:
        List of transforms to apply to the dataset.
    """
    PIL_TRANSFORMS = [
        ("pixel_backdoor", custom_transforms.CornerPixelBackdoor),
    ]
    NP_TRANSFORMS = [
        ("noise", custom_transforms.GaussianNoise),
        ("noise_backdoor", custom_transforms.NoiseBackdoor),
    ]
    transforms: List[Callable] = [
        custom_transforms.AddInfoDict(),
    ]

    def process_transform(name, transform):
        if name in config:
            transform_config = dict(config[name])
            if "enabled" in transform_config:
                if not transform_config["enabled"]:
                    return
                del transform_config["enabled"]
            transforms.append(transform(**transform_config))

    for name, transform in PIL_TRANSFORMS:
        process_transform(name, transform)
    transforms.append(utils.adapt_transform(to_numpy))
    for name, transform in NP_TRANSFORMS:
        process_transform(name, transform)

    return transforms
