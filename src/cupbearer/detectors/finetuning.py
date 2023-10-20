from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from jax import lax
from torch.utils.data import DataLoader

from cupbearer.data import numpy_collate
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.config import DetectorConfig, TrainConfig
from cupbearer.utils import trainer
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig


class FinetuningTrainer(trainer.TrainerModule):
    def __init__(self, model, loss_fn, **kwargs):
        super().__init__(model=model, **kwargs)
        self.loss_fn = loss_fn

    def create_functions(self):
        def train_step(state, batch):
            def loss_fn(params):
                return self.loss_fn(params, batch)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss}
            return state, metrics

        def eval_step(state, batch):
            loss = self.loss_fn(state.params, batch)
            metrics = {"loss": loss}
            return metrics

        return train_step, eval_step


class FinetuningAnomalyDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetuned_params = None

    def train(
        self,
        clean_dataset,
        optimizer: OptimizerConfig,
        num_epochs: int = 10,
        batch_size: int = 128,
        # Not used, but will be passed by the detector train script
        debug: bool = False,
    ):
        trainer_instance = FinetuningTrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=optimizer.build(),
            log_dir=self.save_path,
            override_variables={
                "params": self.params
            },  # Use existing parameters for initialization
            # clean_dataset[0] is the first element, which is an (image, label) tuple.
            # The second 0 picks the image, then we add a singleton batch dimension.
            example_input=clean_dataset[0][0][None, ...],
        )

        # Create a DataLoader for the clean dataset
        clean_loader = DataLoader(
            dataset=clean_dataset,
            batch_size=batch_size,
            collate_fn=numpy_collate,
        )

        # Finetune the model on the clean dataset
        trainer_instance.train_model(
            train_loader=clean_loader,
            val_loaders={},
            num_epochs=num_epochs,
        )

        # Store the finetuned parameters
        self.finetuned_params = trainer_instance.state.params

    def loss_fn(self, params, batch):
        inputs, targets = batch
        logits = self.model.apply({"params": params}, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    def layerwise_scores(self, batch):
        raise NotImplementedError(
            "Layerwise scores don't exist for finetuning detector"
        )

    def scores(self, batch):
        original_output, _ = self._model(batch)

        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        finetuned_output = self.model.apply({"params": self.finetuned_params}, inputs)

        finetuned_p = jax.nn.softmax(finetuned_output, axis=-1)
        original_p = jax.nn.softmax(original_output, axis=-1)

        # Check whether we're going to get infinities:
        if jnp.any(jnp.logical_and(finetuned_p == 0, original_p > 0)):
            # We'd get an error anyway once we compute eval metrics, better to give
            # a more specific one here.
            raise ValueError("Infinite KL divergence")

        # This is the same direction of KL divergence that Redwood used in one of their
        # projects, though I don't know if they had a strong reason for it.
        # Arguably a symmetric metric would make more sense, but might not matter much.
        p, q = original_p, finetuned_p
        # Adapted from jax.scipy.special.rel_entr in Jax >= 0.4.16
        both_gt_zero_mask = lax.bitwise_and((p > 0), (q > 0))
        one_zero_mask = lax.bitwise_and((p == 0), (q >= 0))

        safe_p = jnp.where(both_gt_zero_mask, p, 1)
        safe_q = jnp.where(both_gt_zero_mask, q, 1)
        log_val = lax.sub(
            jax.scipy.special.xlogy(safe_p, safe_p),
            jax.scipy.special.xlogy(safe_p, safe_q),
        )
        kl = jnp.where(
            both_gt_zero_mask, log_val, jnp.where(one_zero_mask, q, jnp.inf)
        ).sum(-1)
        return kl

    def _get_trained_variables(self, saving: bool = False):
        return {
            "params": self.finetuned_params,
        }

    def _set_trained_variables(self, variables):
        self.finetuned_params = variables["params"]


@dataclass
class FinetuningTrainConfig(TrainConfig):
    optimizer: OptimizerConfig = config_group(OptimizerConfig, Adam)
    num_epochs: int = 10
    batch_size: int = 128

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.batch_size = 2


@dataclass
class FinetuningConfig(DetectorConfig):
    train: FinetuningTrainConfig = field(default_factory=FinetuningTrainConfig)

    def build(self, model, params, rng, save_dir) -> FinetuningAnomalyDetector:
        return FinetuningAnomalyDetector(
            model=model,
            params=params,
            rng=rng,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
