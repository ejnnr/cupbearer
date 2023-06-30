from functools import partial
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import hydra
from hydra.utils import to_absolute_path
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import optax
from torch.utils.data import Dataset, DataLoader

from abstractions import abstraction, data, utils


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
            cfg = hydra.compose(
                config_name="adversarial_examples",
                overrides=[f"base_run={base_run}"],
            )
            attack(cfg)
            self.examples = utils.load(base_run / "adv_examples")
        if num_examples is None:
            num_examples = len(self.examples)
        self.num_examples = num_examples
        if len(self.examples) < num_examples:
            raise ValueError(
                f"Only {len(self.examples)} adversarial examples exist, but {num_examples} were requested"
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


@hydra.main(version_base=None, config_path="conf", config_name="adversarial_examples")
def attack(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    # Load the model to attack
    base_run = Path(cfg.base_run)

    if os.path.exists(base_run / f"adv_examples.{utils.SUFFIX}"):
        logger.info("Adversarial examples already exist, skipping attack")
        return

    base_cfg = OmegaConf.load(base_run / ".hydra" / "config.yaml")

    computation = hydra.utils.call(base_cfg.model)
    model = abstraction.Model(computation=computation)
    params = utils.load(base_run / "model")["params"]

    dataset = data.get_dataset(base_cfg.train_data)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    adv_examples = []
    num_examples = 0

    mean_original_loss = 0
    mean_new_loss = 0

    for i, batch in enumerate(dataloader):
        inputs, labels, infos = batch
        adv_inputs, original_loss = fgsm(
            forward_fn=lambda x: model.apply({"params": params}, x),
            inputs=inputs,
            labels=labels,
            eps=cfg.eps,
        )
        adv_examples.append(adv_inputs)
        num_examples += len(adv_inputs)

        new_logits = model.apply({"params": params}, adv_inputs)
        one_hot = jax.nn.one_hot(labels, new_logits.shape[-1])
        new_loss = optax.softmax_cross_entropy(logits=new_logits, labels=one_hot).mean()
        logger.log("METRICS", f"original loss={original_loss}, new loss={new_loss}")
        mean_original_loss = (i * mean_original_loss + original_loss) / (i + 1)
        mean_new_loss += (i * mean_new_loss + new_loss) / (i + 1)

        if num_examples >= cfg.num_examples:
            break

    adv_examples = jnp.concatenate(adv_examples, axis=0)
    utils.save(adv_examples, base_run / "adv_examples")
    with open(base_run / "adv_examples.json", "w") as f:
        json.dump(
            {
                "original_loss": mean_original_loss,
                "new_loss": mean_new_loss,
                "eps": cfg.eps,
                "num_examples": num_examples,
            },
            f,
        )

    # Plot a few adversarial examples in a grid and save the plot as a pdf
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax = axs[i // 3, i % 3]
        ax.imshow(adv_examples[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(base_run / "adv_examples.pdf")


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, icon="📈")
    logger.add(
        sys.stdout, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stdout, filter=lambda record: record["level"].name != "METRICS")
    attack()
