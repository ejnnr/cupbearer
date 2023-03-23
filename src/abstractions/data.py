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


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def to_numpy(img):
    return np.array(img, dtype=jnp.float32) / 255.0


def get_data_loaders(batch_size, p_backdoor=0.5):
    """Load MNIST train and test datasets into memory."""
    # Compose is meant to just compose image transforms, rather than
    # the joint transforms we have here. But the implementation is
    # actually agnostic as to whether the sample is just an image
    # or a tuple with multiple elements.
    transforms = Compose(
        [
            backdoor.CornerPixelToWhite(p_backdoor),
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

    train_loader = NumpyLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = NumpyLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader
