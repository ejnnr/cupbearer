import os
import sys
import hydra
import jax
import jax.numpy as jnp

from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
import optax
from loguru import logger

from abstractions import abstraction, data, trainer, utils
from abstractions.logger import DummyLogger, WandbLogger


class ClassificationTrainer(trainer.TrainerModule):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def create_functions(self):
        def losses(params, state, batch):
            images, labels, infos = batch
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, self.num_classes)
            losses = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
            loss = jnp.mean(losses)
            correct = jnp.argmax(logits, -1) == labels
            accuracy = jnp.mean(correct)

            return loss, accuracy

        def train_step(state, batch):
            loss_fn = lambda params: losses(params, state, batch)
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            loss, accuracy = losses(state.params, state, batch)  # type: ignore

            state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss, "accuracy": accuracy}
            return state, metrics

        def eval_step(state, batch):
            loss, accuracy = losses(state.params, state, batch)  # type: ignore
            metrics = {"loss": loss, "accuracy": accuracy}
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


@hydra.main(version_base=None, config_path="conf", config_name="train_base")
def main(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            tags=["base-training-backdoor"],
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    dataset = data.get_dataset(cfg.train_data)
    train_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=data.numpy_collate
    )

    val_loaders = {}
    for k, v in cfg.val_data.items():
        dataset = data.get_dataset(v)
        val_loaders[k] = DataLoader(
            dataset,
            batch_size=cfg.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )

    # Dataloader returns images, labels and infos, only images get passed to model
    images, _, _ = next(iter(train_loader))
    example_input = images[0:1]

    computation = hydra.utils.call(cfg.model)
    model = abstraction.Model(computation)

    trainer = ClassificationTrainer(
        num_classes=cfg.num_classes,
        model=model,
        optimizer=hydra.utils.instantiate(cfg.optim),
        example_input=example_input,
        # Hydra sets the cwd to the right log dir automatically
        log_dir=".",
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )

    trainer.train_model(
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=None,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
    )
    trainer.save_model()
    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, icon="📈")
    logger.add(
        sys.stdout, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stdout, filter=lambda record: record["level"].name != "METRICS")
    # We want to escape slashes in arguments that get reused as filenames.
    utils.register_resolvers()
    main()
