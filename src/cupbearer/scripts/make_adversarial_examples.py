import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from cupbearer.data import TrainDataFromRun, numpy_collate
from cupbearer.data.adversarial import fgsm
from cupbearer.models import StoredModel
from cupbearer.scripts.conf.make_adversarial_examples_conf import Config
from cupbearer.utils import utils
from cupbearer.utils.scripts import run
from loguru import logger
from torch.utils.data import DataLoader


def attack(cfg: Config):
    if cfg.dir.path is None:
        raise ValueError("Must specify a run path")

    if os.path.exists(cfg.dir.path / f"adv_examples.{utils.SUFFIX}"):
        logger.info("Adversarial examples already exist, skipping attack")
        return

    model_cfg = StoredModel(path=cfg.dir.path)
    model = model_cfg.build_model()
    params = model_cfg.build_params()

    data_cfg = TrainDataFromRun(path=cfg.dir.path)
    dataset = data_cfg.build()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )

    adv_examples = []
    labels = []
    num_examples = 0

    mean_original_loss = 0
    mean_new_loss = 0
    mean_new_accuracy = 0

    for i, batch in enumerate(dataloader):
        inputs, new_labels = batch
        adv_inputs, original_loss = fgsm(
            forward_fn=lambda x: model.apply({"params": params}, x),
            inputs=inputs,
            labels=new_labels,
            eps=cfg.eps,
        )
        # FGSM might have given us pixel values that don't actually correspond to colors
        adv_inputs = jnp.clip(adv_inputs, 0, 1)
        adv_examples.append(adv_inputs)
        labels.append(new_labels)
        num_examples += len(adv_inputs)

        new_logits = model.apply({"params": params}, adv_inputs)
        one_hot = jax.nn.one_hot(new_labels, new_logits.shape[-1])
        new_accuracy = jnp.mean(jnp.argmax(new_logits, -1) == new_labels)
        new_loss = optax.softmax_cross_entropy(logits=new_logits, labels=one_hot).mean()
        logger.info(f"original loss={original_loss}, new loss={new_loss}")
        mean_original_loss = (i * mean_original_loss + original_loss) / (i + 1)
        mean_new_loss = (i * mean_new_loss + new_loss) / (i + 1)
        mean_new_accuracy = (i * mean_new_accuracy + new_accuracy) / (i + 1)

        if cfg.max_examples and num_examples >= cfg.max_examples:
            break

    if mean_new_accuracy > cfg.success_threshold:
        raise RuntimeError(
            "Attack failed, new accuracy is "
            f"{mean_new_accuracy} > {cfg.success_threshold}."
        )

    adv_examples = jnp.concatenate(adv_examples, axis=0)
    labels = jnp.concatenate(labels, axis=0)
    # Need to wrap the array in a container to make the orbax checkpointer work.
    utils.save(
        {"adv_examples": adv_examples, "labels": labels}, cfg.dir.path / "adv_examples"
    )
    with open(cfg.dir.path / "adv_examples.json", "w") as f:
        json.dump(
            {
                "original_loss": mean_original_loss.item(),
                "new_loss": mean_new_loss.item(),
                "new_accuracy": mean_new_accuracy.item(),
                "eps": cfg.eps,
                "num_examples": num_examples,
            },
            f,
        )

    # Plot a few adversarial examples in a grid and save the plot as a pdf
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax = axs[i // 3, i % 3]
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.imshow(adv_examples[i])
        except IndexError:
            pass
    plt.tight_layout()
    plt.savefig(cfg.dir.path / "adv_examples.pdf")


if __name__ == "__main__":
    run(attack, Config)
