# Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html

# Standard libraries
from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
    Optional,
    Tuple,
    Dict,
    Callable,
    Union,
)
import json
import time
from loguru import logger
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict

# JAX/Flax
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax import struct, core
from flax.training import train_state, checkpoints
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

from hydra.utils import to_absolute_path
from abstractions import utils

from abstractions.utils import SizedIterable, original_relative_path, save, load
from abstractions.logger import Logger


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


class InferenceState(struct.PyTreeNode):
    # Lightweight alternative to TrainState that doesn't include the optimizer
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    batch_stats: Any = None
    rng: Any = None


class TrainerModule(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optax.GradientTransformation],
        example_input: Any,
        loggers: Iterable[Logger],
        log_dir: str = "logs",
        seed: int = 42,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 1,
        **kwargs,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model: The instantiated model to train.
          optimizer: The instantiated optax optimizer. If not provided,
            training will not be possible.
          example_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.optimizer = optimizer
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.example_input = example_input
        self.loggers = loggers
        self.log_dir = Path(log_dir)
        self.model = model
        # Set of hyperparameters to save
        self.config = {
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
            # We want the actual absolute path, e.g. if log_dir is ".",
            # we want to resolve to the CWD created by Hydra.
            # That makes it easy to load the model later.
            "log_dir": str(self.log_dir.absolute()),
        }
        self.config.update(kwargs)
        self.print_tabulate()
        # Init trainer parts
        self.create_jitted_functions()
        self.init_model(optimizer)

    def init_model(self, optimizer: Optional[optax.GradientTransformation]):
        """
        Creates an initial training state with newly generated network parameters.

        Args:
          optimizer: The instantiated optax optimizer.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        variables = self.run_model_init(init_rng)
        # Create default state
        self.init_state(variables, model_rng, optimizer)

    def init_state(self, variables, rng, tx):
        if tx is None:
            self.state = TrainState(
                step=0,
                apply_fn=self.model.apply,
                params=variables["params"],
                batch_stats=variables.get("batch_stats"),
                rng=rng,
                tx=None,
                opt_state=None,
            )
        else:
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=variables["params"],
                batch_stats=variables.get("batch_stats"),
                rng=rng,
                tx=tx,
            )

    def run_model_init(self, init_rng: Any) -> Dict:
        """
        The model initialization call

        Args:
          example_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        return self.model.init(init_rng, self.example_input, train=True)

    def print_tabulate(self):
        """
        Prints a summary of the Module represented as table.
        """
        # Filter out ANSI escape codes (tabulate uses colors, but these show up as
        # escape codes in Slurm log files)
        print(
            utils.escape_ansi(
                self.model.tabulate(random.PRNGKey(0), self.example_input, train=True)
            )
        )

    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    @abstractmethod
    def create_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Dict],
    ]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """

        def train_step(state: TrainState, batch: Any):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            metrics = {}
            return metrics

        raise NotImplementedError

    def train_model(
        self,
        train_loader: SizedIterable,
        val_loaders: Mapping[str, SizedIterable],
        test_loaders: Optional[Mapping[str, SizedIterable]] = None,
        num_epochs: int = 500,
        max_steps: Optional[int] = None,
    ):
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loaders: Any number of validation loaders (often this will just be
            a single one, but you might have different types of validation sets).
            Key should be a name for each one, e.g. `{"val": val_loader}`.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.
          max_steps: If given, training will stop after this many steps.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        if self.state.tx is None:
            raise ValueError("No optimizer was given. Please specify an optimizer.")
        # Prepare training loop
        metrics = {}
        self.on_training_start()
        for epoch_idx in self.pbar(range(1, num_epochs + 1), desc="Epochs"):
            train_metrics = self.train_epoch(train_loader, max_steps=max_steps)
            metrics[epoch_idx] = train_metrics
            self.on_training_epoch_end(epoch_idx, train_metrics)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loaders)
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loaders)
                self.log_metrics(eval_metrics)
                metrics[epoch_idx].update(eval_metrics)

        # Test model if possible
        if test_loaders is not None:
            test_metrics = self.eval_model(test_loaders)
            self.log_metrics(test_metrics)
            metrics["test"] = test_metrics

        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

    def train_epoch(
        self, train_loader: SizedIterable, max_steps: Optional[int]
    ) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        avg_metrics = defaultdict(float)
        # Train model for one epoch and log metrics
        num_batches = len(train_loader)
        for batch in self.pbar(train_loader, desc="Training", leave=False):
            if max_steps is not None and self.state.step >= max_steps:
                break
            self.state, step_metrics = self.train_step(self.state, batch)
            step_metrics = {k: v.item() for k, v in step_metrics.items()}
            self.log_metrics(
                {"train/" + k: v for k, v in step_metrics.items()},
            )
            for key in step_metrics:
                avg_metrics["train/" + key] += step_metrics[key] / num_batches

        return avg_metrics

    def eval_model(self, data_loaders: Mapping[str, SizedIterable]) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loaders: A dictionary of data loaders, where the key is the name

        Returns:
          A dictionary with metrics, averaged over all batches. Metrics from
          all dataloaders are merged into one dictionary by prefixing the name.
        """
        # Test model on all images of a data loader and return avg loss
        num_elements = 0
        metrics = {}
        for data_loader_name, data_loader in data_loaders.items():
            data_loader_metrics = defaultdict(float)
            for batch in data_loader:
                step_metrics = self.eval_step(self.state, batch)
                step_metrics = {k: v.item() for k, v in step_metrics.items()}
                batch_size = (
                    batch[0].shape[0]
                    if isinstance(batch, (list, tuple))
                    else batch.shape[0]
                )
                for key in step_metrics:
                    data_loader_metrics[key] += step_metrics[key] * batch_size
                num_elements += batch_size

            for key, value in data_loader_metrics.items():
                metrics[data_loader_name + "/" + key] = value / num_elements

        return metrics

    def pbar(self, iterable: Iterable, **kwargs) -> Iterable:
        """
        Wraps an iterable in a progress bar tracker (tqdm) if the progress bar
        is enabled.

        Args:
          iterable: Iterable to wrap in tqdm.
          kwargs: Additional arguments to tqdm.

        Returns:
          Wrapped iterable if progress bar is enabled, otherwise same iterable
          as input.
        """
        if self.enable_progress_bar:
            return tqdm(iterable, **kwargs)
        else:
            return iterable

    def log_metrics(self, metrics: Mapping[str, Any]):
        for logger in self.loggers:
            logger.log_metrics(metrics, step=int(self.state.step))

    def close_loggers(self):
        for logger in self.loggers:
            logger.close()

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(
        self,
        epoch_idx: int,
        metrics: Dict[str, Any],
        val_loaders: Mapping[str, SizedIterable],
    ):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loaders: Data loaders of all validation sets, to support additional
            evaluation.
        """
        pass

    def save_model(self):
        """
        Saves current training state at certain training iteration. Only the model
        parameters and batch statistics are saved to reduce memory footprint. To
        support the training to be continued from a checkpoint, this method can be
        extended to include the optimizer state as well.
        """
        # Save hyperparameters
        if os.path.isfile(self.log_dir / "hparams.json"):
            with open(self.log_dir / "hparams.json", "r") as f:
                old_config = json.load(f)
            if old_config != self.config:
                raise RuntimeError(
                    f"Hyperparameters in {self.log_dir/ 'hparams.json'} have changed. "
                    "Please save model to a different directory."
                )

        with open(self.log_dir / "hparams.json", "w") as f:
            json.dump(self.config, f, indent=4)

        # Save model parameters and batch statistics
        save(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            self.log_dir / "model",
        )
        logger.info(f"Saved model to {original_relative_path(self.log_dir / 'model')}")

    def load_model(self):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        state_dict = load(self.log_dir / "model")
        self.init_state(variables=state_dict, rng=self.state.rng, tx=self.state.tx)

    def bind_model(self):
        """
        Returns a model with parameters bound to it. Enables an easier inference
        access.

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {"params": self.state.params}
        if self.state.batch_stats:
            params["batch_stats"] = self.state.batch_stats
        return self.model.bind(params)
