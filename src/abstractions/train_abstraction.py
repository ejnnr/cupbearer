# based on the flax example code

from collections import defaultdict
import sys
from typing import Sequence
import jax
import jax.numpy as jnp
from jax.config import config as jax_config
import numpy as np
from flax import linen as nn

from flax.training import train_state
import optax
from loguru import logger
import argparse
from clearml import Task

from abstractions import abstraction, data, train_mnist, utils, trainer
from abstractions.logger import ClearMLLogger, DummyLogger


class AbstractionTrainer(trainer.TrainerModule):
    def __init__(self, abstract_dim: int, output_dim: int, **kwargs):
        super().__init__(
            model_class=abstraction.Abstraction,  # type: ignore
            model_hparams={"abstract_dim": abstract_dim, "output_dim": output_dim},
            **kwargs,
        )

    def create_functions(self):
        def losses(params, state, batch, mask=None):
            logits, activations = batch
            abstractions, predicted_abstractions, predicted_logits = state.apply_fn(
                {"params": params}, activations
            )
            assert isinstance(abstractions, list)
            assert isinstance(predicted_abstractions, list)
            assert len(abstractions) == len(predicted_abstractions) + 1
            b, d = abstractions[0].shape
            assert predicted_abstractions[0].shape == (b, d)
            assert logits.shape == (b, 10) == predicted_logits.shape

            if mask is None:
                mask = jnp.ones((b,))
            assert mask.shape == (b,)
            num_samples = mask.sum()

            # Output loss (KL divergence between actual and predicted output):
            # TODO: Should probably just use something like distrax.
            probs = jax.nn.softmax(logits, axis=-1)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            predicted_logprobs = jax.nn.log_softmax(predicted_logits, axis=-1)
            output_losses = (probs * (logprobs - predicted_logprobs)).sum(axis=-1)
            output_losses *= mask
            # Can't take mean here because we don't want to count masked samples
            output_loss = output_losses.sum() / num_samples
            # Consistency loss:
            consistency_loss = jnp.array(0)
            # Skip the first abstraction, since there's no prediction for that
            for abstraction, predicted_abstraction in zip(
                abstractions[1:], predicted_abstractions
            ):
                # Take mean over hidden dimension:
                consistency_losses = ((abstraction - predicted_abstraction) ** 2).mean(
                    -1
                )
                if mask is not None:
                    consistency_losses *= mask
                # Now we also take the mean over the batch (outside the sqrt and after
                # masking). As before, we don't want to count masked samples.
                consistency_loss += jnp.sqrt(consistency_losses).sum() / num_samples

            consistency_loss /= len(predicted_abstractions)

            loss = output_loss + consistency_loss

            return loss, (output_loss, consistency_loss)

        def train_step(state, batch):
            loss_fn = lambda params: losses(params, state, batch)
            (loss, (output_loss, consistency_loss)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
            param_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), state.params)
            max_param_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(param_norms)))

            grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
            max_grad_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(grad_norms)))

            state = state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
                "max_param_norm": max_param_norm,
                "max_grad_norm": max_grad_norm,
            }
            return state, metrics

        def eval_step(state, batch):
            # Note that this class expects the eval dataloader to return not just
            # logits/activations, but also the original data. That's necessary
            # to compute metrics on specific clases (zero in this case).
            logits, activations, (images, labels, infos) = batch
            loss, (output_loss, consistency_loss) = losses(
                state.params, state, (logits, activations)
            )
            zeros = infos["original_target"] == 0
            zeros_loss, (zeros_output_loss, zeros_consistency_loss) = losses(
                state.params, state, (logits, activations), mask=zeros
            )
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
                "zeros_loss": zeros_loss,
                "zeros_output_loss": zeros_output_loss,
                "zeros_consistency_loss": zeros_consistency_loss,
            }
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def train_and_evaluate(config):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
    """
    if config.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)
        config.no_clearml = True

    if config.no_clearml:
        metrics_logger = DummyLogger()
    else:
        metrics_logger = ClearMLLogger(
            project_name="backdoor-detection", task_name="train MNIST abstraction"
        )

    # Load the full model we want to abstract
    model = train_mnist.MLP()
    params = utils.load(config.model_path)

    # Magic collate_fn to get the activations of the model
    train_collate_fn = abstraction.abstraction_collate(model, params)
    val_collate_fn = abstraction.abstraction_collate(
        model, params, return_original_batch=True
    )

    train_loader, _ = data.get_data_loaders(
        config.batch_size, p_backdoor=0.0, collate_fn=train_collate_fn
    )
    # For validation, we still use the training data, but with backdoors.
    # TODO: this doesn't feel very elegant.
    # Need to think about what's the principled thing to do here.
    backdoor_loader, _ = data.get_data_loaders(
        config.batch_size, p_backdoor=1.0, collate_fn=val_collate_fn
    )
    different_corner_loader, _ = data.get_data_loaders(
        config.batch_size, p_backdoor=1.0, collate_fn=val_collate_fn, corner="top-right"
    )
    val_loaders = {
        "backdoor": backdoor_loader,
        "different_corner": different_corner_loader,
    }

    # Dataloader returns logits and activations, only activations get passed to model
    _, example_activations = next(iter(train_loader))
    # Activations are a list of batched activations, we want to effectively get
    # batch size 1
    example_input = [x[0:1] for x in example_activations]

    trainer = AbstractionTrainer(
        abstract_dim=config.abstract_dim,
        output_dim=10,
        optimizer_hparams={
            "lr": config.learning_rate,
            "optimizer": config.optimizer,
        },
        example_input=example_input,
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )

    trainer.train_model(train_loader, val_loaders, num_epochs=config.num_epochs)
    trainer.close_loggers()


def parse_args():
    parser = argparse.ArgumentParser(description="Jax MNIST training example")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--abstract_dim", type=int, default=256, help="Abstract dim")
    parser.add_argument("--model_path", type=str, help="Path to model", required=True)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--no_clearml", action="store_true", help="Disable ClearML")
    parser.add_argument(
        "--workdir", type=str, default="logs", help="Directory for logs"
    )
    return parser.parse_args()


def main():
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    config = parse_args()
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
