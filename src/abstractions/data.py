from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import jax.numpy as jnp

from abstractions import backdoor, utils


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
    return np.array(img, dtype=jnp.float32) / 255.0


def get_data_loaders(
    batch_size, collate_fn=numpy_collate, **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST train and test datasets into memory.

    Args:
        batch_size: Batch size for the data loaders.
        collate_fn: collate_fn for pytorch DataLoader.
        **kwargs: Additional keyword arguments to pass to CornerPixelToWhite.

    Returns:
        Tuple (train_loader, test_loader)
    """
    # Compose is meant to just compose image transforms, rather than
    # the joint transforms we have here. But the implementation is
    # actually agnostic as to whether the sample is just an image
    # or a tuple with multiple elements.
    transforms = Compose(
        [
            backdoor.CornerPixelToWhite(**kwargs),
            utils.adapt_transform(to_numpy),
        ]
    )
    CustomMNIST = utils.add_transforms(MNIST)
    train_dataset = CustomMNIST(
        root="data", train=True, transforms=transforms, download=True
    )
    test_dataset = CustomMNIST(
        root="data", train=False, transforms=transforms, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader
