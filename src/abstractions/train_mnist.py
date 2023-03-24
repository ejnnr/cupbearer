# based on the flax example code

from collections import defaultdict
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from flax.training import train_state
import optax
from loguru import logger
import argparse
from clearml import Task

from abstractions import data, utils


class MLP(nn.Module):
    """A simple feed-forward MLP."""

    @nn.compact
    def __call__(self, x, return_activations=False):
        activations = []
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        activations.append(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        activations.append(x)
        x = nn.Dense(features=10)(x)

        if return_activations:
            return x, activations
        return x


@jax.jit
def apply_model(state, batch):
    """Computes gradients, loss and metrics for a single batch."""

    images, labels, backdoored = batch

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    correct = jnp.argmax(logits, -1) == labels
    num_backdoored = jnp.sum(backdoored)
    # Using this slightly convoluted way of computing accuracy on subsets
    # to make everything jit-able.
    backdoor_accuracy = jnp.sum(correct * backdoored) / num_backdoored
    non_backdoor_accuracy = jnp.sum(correct * (1 - backdoored)) / (
        len(backdoored) - num_backdoored
    )
    accuracy = jnp.mean(correct)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "backdoor_accuracy": backdoor_accuracy,
        "non_backdoor_accuracy": non_backdoor_accuracy,
    }
    return grads, metrics


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_loader, rng, metrics_logger):
    """Train for a single epoch."""
    train_ds_size = len(train_loader.dataset)
    steps_per_epoch = train_ds_size // train_loader.batch_size

    epoch_metrics = defaultdict(list)

    for batch in train_loader:
        grads, metrics = apply_model(state, batch)
        state = update_model(state, grads)
        for k, v in metrics.items():
            epoch_metrics[k].append(v)
            metrics_logger.report_scalar(
                title="Training", series=k, value=v.item(), iteration=int(state.step)
            )

    train_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
    return state, train_metrics


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model = MLP()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(config) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.

    Returns:
      The train state (which includes the `.params`).
    """
    # seeds pytorch and numpy
    Task.set_random_seed(0)
    task = Task.init(
        project_name="backdoor-detection", task_name="train MNIST backdoor"
    )
    metrics_logger = task.get_logger()

    train_loader, test_loader = data.get_data_loaders(config.batch_size)
    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        metrics_logger.report_scalar("epoch", "epoch", epoch, int(state.step))
        state, train_metrics = train_epoch(
            state, train_loader, input_rng, metrics_logger
        )
        test_batch = next(iter(test_loader))
        _, test_metrics = apply_model(state, test_batch)

        logger.log(
            "METRICS",
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"] * 100,
                test_metrics["loss"],
                test_metrics["accuracy"] * 100,
            ),
        )

    if config.save_path:
        utils.save(state.params, config.save_path)
    else:
        utils.save(state.params, "models/mnist", overwrite=True)

    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Jax MNIST training example")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--save_path", type=str, default=None)
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
