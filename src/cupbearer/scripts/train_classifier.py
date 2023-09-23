import dataclasses

import jax
import jax.numpy as jnp
import optax
from cupbearer.data import numpy_collate
from cupbearer.utils import trainer
from cupbearer.utils.logger import DummyLogger, WandbLogger
from cupbearer.utils.scripts import run
from loguru import logger
from torch.utils.data import DataLoader

from .conf.train_classifier_conf import Config


class ClassificationTrainer(trainer.TrainerModule):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def create_functions(self):
        def losses(params, state, batch):
            images, labels = batch
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, self.num_classes)
            losses = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
            loss = jnp.mean(losses)
            correct = jnp.argmax(logits, -1) == labels
            accuracy = jnp.mean(correct)

            return loss, accuracy

        def train_step(state, batch):
            def loss_fn(params):
                return losses(params, state, batch)

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
        logger.info(self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.info(self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def main(cfg: Config):
    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            tags=["base-training-backdoor"],
            config=dataclasses.asdict(cfg),
        )
    else:
        metrics_logger = DummyLogger()

    dataset = cfg.train_data.build()
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
    )

    val_loaders = {}
    for k, v in cfg.val_data.items():
        dataset = v.build()
        val_loaders[k] = DataLoader(
            dataset,
            batch_size=cfg.max_batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )

    # Dataloader returns images and labels, only images get passed to model
    images, _ = next(iter(train_loader))
    example_input = images[0:1]

    model = cfg.model.build_model()

    trainer = ClassificationTrainer(
        num_classes=cfg.num_classes,
        model=model,
        optimizer=cfg.optim.build(),
        example_input=example_input,
        log_dir=cfg.dir.path,
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    trainer.train_model(
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=None,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
    )
    if cfg.dir.path:
        trainer.save_model()
        for trafo in cfg.train_data.get_transforms():
            trafo.store(cfg.dir.path)
    trainer.close_loggers()


if __name__ == "__main__":
    run(main, Config)
