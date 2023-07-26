from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
from functools import partial
from abstractions.utils import utils
from abstractions.utils.hydra import hydra_config
from . import DatasetConfig

from loguru import logger
from torch.utils.data import Dataset

import subprocess
import sys
from pathlib import Path


@hydra_config
@dataclass
class AdversarialExampleConfig(DatasetConfig):
    run_path: str

    def _get_dataset(self) -> Dataset:
        return AdversarialExampleDataset(
            base_run=self.run_path, num_examples=self.max_size
        )


class AdversarialExampleDataset(Dataset):
    def __init__(self, base_run, num_examples=None):
        base_run = Path(base_run)
        self.base_run = base_run
        try:
            self.examples = utils.load(base_run / "adv_examples")
        except FileNotFoundError:
            logger.info(
                "Adversarial examples not found, running attack with default settings"
            )
            # Calling the hydra.main function directly within an existing hydra job
            # is pretty fiddly, so we just run it as a suprocess.
            subprocess.run(
                [
                    "python",
                    "-m",
                    "abstractions.scripts.adversarial_examples",
                    # Need to quote base_run because it might contain commas
                    f"base_run='{base_run}'",
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
            self.examples = utils.load(base_run / "adv_examples")
        if num_examples is None:
            num_examples = len(self.examples)
        self.num_examples = num_examples
        if len(self.examples) < num_examples:
            raise ValueError(
                f"Only {len(self.examples)} adversarial examples exist, "
                f"but {num_examples} were requested"
            )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples:
            raise IndexError(f"Index {idx} is out of range")
        return self.examples[idx]


@partial(jax.jit, static_argnames=("forward_fn", "eps"))
def fgsm(forward_fn, inputs, labels, eps=8 / 255):
    def loss(x):
        logits = forward_fn(x)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        losses = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        return jnp.mean(losses)

    loss, grad = jax.value_and_grad(loss)(inputs)
    return inputs + eps * jnp.sign(grad), loss
