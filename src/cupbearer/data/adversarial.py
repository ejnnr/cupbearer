import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from loguru import logger
from torch.utils.data import Dataset

from cupbearer.utils import utils

from . import DatasetConfig


@dataclass
class AdversarialExampleConfig(DatasetConfig):
    run_path: Path
    attack_batch_size: Optional[int] = None
    success_threshold: float = 0.1

    def _build(self) -> Dataset:
        return AdversarialExampleDataset(
            base_run=self.run_path,
            num_examples=self.max_size,
            attack_batch_size=self.attack_batch_size,
            success_threshold=self.success_threshold,
        )

    def _set_debug(self):
        super()._set_debug()
        self.attack_batch_size = 2
        self.success_threshold = 1.0


class AdversarialExampleDataset(Dataset):
    def __init__(
        self,
        base_run,
        num_examples=None,
        attack_batch_size: Optional[int] = None,
        success_threshold: float = 0.1,
    ):
        base_run = Path(base_run)
        self.base_run = base_run
        try:
            data = utils.load(base_run / "adv_examples")
            self.examples = data["adv_examples"]
            self.labels = data["labels"]
        except FileNotFoundError:
            logger.info(
                "Adversarial examples not found, running attack with default settings"
            )
            command = [
                "python",
                "-m",
                "cupbearer.scripts.make_adversarial_examples",
                "--dir.full",
                str(base_run),
                "--success_threshold",
                str(success_threshold),
            ]
            if num_examples is not None:
                command += ["--max_examples", str(num_examples)]
            if attack_batch_size is not None:
                command += ["--batch_size", str(attack_batch_size)]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(str(e.stdout).replace("\\n", "\n"))
                print(str(e.stderr).replace("\\n", "\n"))
                raise e
            else:
                print(result.stdout)
                print(result.stderr)

            data = utils.load(base_run / "adv_examples")
            self.examples = data["adv_examples"]
            self.labels = data["labels"]

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
        # Labels are the original ones. We need to return them mainly for implementation
        # reasons: for eval, normal and anomalous data will be batched together, so
        # since the normal data includes labels, the anomalous one needs to as well.
        # TODO: Probably detectors should just never have access to labels during evals
        # (none of the current ones make use of them anyway). If a detector needs them,
        # it should use the model-generated labels, not ground truth ones.
        return self.examples[idx], self.labels[idx]


@partial(jax.jit, static_argnames=("forward_fn", "eps"))
def fgsm(forward_fn, inputs, labels, eps=8 / 255):
    def loss(x):
        logits = forward_fn(x)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        losses = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        return jnp.mean(losses)

    loss, grad = jax.value_and_grad(loss)(inputs)
    return inputs + eps * jnp.sign(grad), loss
