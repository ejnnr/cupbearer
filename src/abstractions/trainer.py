# Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html

# Standard libraries
from abc import ABC, abstractmethod
import os
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
from flax.training import train_state, checkpoints
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

from clearml import Task

from abstractions.utils import SizedIterable
from abstractions.logger import Logger


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


class TrainerModule(ABC):
    def __init__(
        self,
        model_class: nn.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
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
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          example_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        if not isinstance(example_input, (list, tuple)):
            example_input = [example_input]
        self.example_input = example_input
        self.loggers = loggers
        self.log_dir = log_dir
        # Set of hyperparameters to save
        self.config = {
            "model_class": model_class.__name__,
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
        }
        self.config.update(kwargs)
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.print_tabulate()
        # Init trainer parts
        self.create_jitted_functions()
        self.init_model()

    def init_model(self):
        """
        Creates an initial training state with newly generated network parameters.

        Args:
          example_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        variables = self.run_model_init(init_rng)
        # Create default state. Optimizer is initialized later
        self.state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            rng=model_rng,
            tx=None,
            opt_state=None,
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
        print(self.model.tabulate(random.PRNGKey(0), self.example_input, train=True))

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{optimizer_name}"'
        # Initialize learning rate scheduler
        lr = hparams.pop("lr", 1e-3)
        optimizer = opt_class(learning_rate=lr, **hparams)
        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            batch_stats=self.state.batch_stats,
            tx=optimizer,
            rng=self.state.rng,
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
        test_loader: Optional[SizedIterable] = None,
        num_epochs: int = 500,
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

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()
        for epoch_idx in self.pbar(range(1, num_epochs + 1), desc="Epochs"):
            train_metrics = self.train_epoch(train_loader)
            self.on_training_epoch_end(epoch_idx, train_metrics)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loaders)
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loaders)
                self.log_metrics(eval_metrics, step=int(self.state.step))

    def train_epoch(self, train_loader: SizedIterable) -> Dict[str, Any]:
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
            self.state, step_metrics = self.train_step(self.state, batch)
            step_metrics = {k: v.item() for k, v in step_metrics.items()}
            self.log_metrics(
                {"train/" + k: v for k, v in step_metrics.items()},
                step=int(self.state.step),
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

    def log_metrics(self, metrics: Mapping[str, Any], step: int):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

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

    def save_model(self, step: int = 0):
        """
        Saves current training state at certain training iteration. Only the model
        parameters and batch statistics are saved to reduce memory footprint. To
        support the training to be continued from a checkpoint, this method can be
        extended to include the optimizer state as well.

        Args:
          step: Index of the step to save the model at, e.g. epoch.
        """
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target={"params": self.state.params, "batch_stats": self.state.batch_stats},
            step=step,
            overwrite=True,
        )

    def load_model(self):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict["params"],
            batch_stats=state_dict["batch_stats"],
            # Optimizer will be overwritten when training starts
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
            rng=self.state.rng,
        )

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

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, example_input: Any) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          example_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file"
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        hparams.pop("model_class")
        hparams.update(hparams.pop("model_hparams"))
        # TODO: also restore loggers (e.g. by passing the class and hparams
        # instead of the object in __init__)
        hparams["loggers"] = []
        trainer = cls(example_input=example_input, **hparams)
        trainer.load_model()
        return trainer