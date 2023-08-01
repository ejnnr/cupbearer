from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import (
    FilteredAbstraction,
    abstraction_collate,
)
from cupbearer.models.computations import Step
from cupbearer.utils import trainer, utils
from cupbearer.utils.trainer import SizedIterable

from .abstraction_detector import OUTPUT_LOSS_FNS, AbstractionDetector, compute_losses


class AbstractionFinetuner(trainer.TrainerModule):
    def __init__(
        self,
        output_loss_fn: str = "kl",
        normal_weight: float = 0.5,
        clip: bool = True,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.output_loss_fn = OUTPUT_LOSS_FNS[output_loss_fn]
        self.normal_weight = normal_weight
        self.clip = clip

    def on_training_start(self, train_loader: SizedIterable):
        if self.clip:
            loss = 0
            for batch in train_loader:
                new_loss, _ = compute_losses(
                    self.state.params,
                    self.state,
                    batch["normal"],
                    output_loss_fn=self.output_loss_fn,
                )
                loss += new_loss
            loss /= len(train_loader)
            self.initial_loss = loss
            logger.info(f"Clipping normal loss to {loss:.4f}")

    def create_functions(self):
        def train_step(state, batch):
            normal_batch, new_batch = batch["normal"], batch["new"]

            def loss_fn(params, batch, normal):
                loss, (_, _, norm) = compute_losses(
                    params, state, batch, output_loss_fn=self.output_loss_fn
                )
                if self.clip and normal:
                    # We don't want to incentivize getting the loss below the initial
                    # loss, just want to make sure it doesn't exceed that
                    loss = jnp.clip(loss, a_min=self.initial_loss)
                    # losses = jax.tree_map(
                    #     lambda loss: jnp.clip(loss, a_min=self.initial_loss), losses
                    # )
                return loss, norm

            (normal_loss, normal_norm), normal_grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params, normal_batch, normal=True)

            (new_loss, new_norm), new_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, new_batch, normal=False
            )

            keys = set(normal_grads.keys())
            assert keys == set(new_grads.keys())
            assert keys <= {"tau_maps", "computational_steps", "filter_maps"}, keys

            # For the computational steps, we want to minimize both losses, so just
            # add the normal gradients. For the tau maps, we want to maximize the loss
            # on new data, so we subtract the new gradients instead.
            grads = {
                "computational_steps": utils.weighted_sum(
                    normal_grads["computational_steps"],
                    new_grads["computational_steps"],
                    self.normal_weight,
                ),
                "tau_maps": utils.weighted_sum(
                    normal_grads["tau_maps"],
                    utils.negative(new_grads["tau_maps"]),
                    self.normal_weight,
                ),
            }

            if "filter_maps" in keys:
                grads["filter_maps"] = utils.weighted_sum(
                    normal_grads["filter_maps"],
                    new_grads["filter_maps"],
                    self.normal_weight,
                )

            grads = FrozenDict(grads)

            state = state.apply_gradients(grads=grads)
            metrics = {
                "normal_loss": normal_loss,
                "new_loss": new_loss,
                "normal_norm": normal_norm,
                "new_norm": new_norm,
            }
            return state, metrics

        def eval_step(state, batch):
            return {}

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.info(self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


class AdversarialAbstractionDetector(AbstractionDetector):
    def __init__(
        self,
        normal_dataset: Dataset,
        filter_maps: Optional[list[Step]] = None,
        new_batch_size: int = 128,
        normal_batch_size: int = 128,
        num_epochs: int = 1,
        normal_weight: float = 0.5,
        clip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normal_dataset = normal_dataset
        self.filter_maps = filter_maps
        self.new_batch_size = new_batch_size
        self.normal_batch_size = normal_batch_size
        self.num_epochs = num_epochs
        self.normal_weight = normal_weight
        self.clip = clip

    def _finetune(
        self,
        new_dataset,
    ) -> dict:
        # TODO: a lot of this can be done once in init instead of every time
        # (though performance overhead is probably small compared to training itself)
        assert self.abstraction is not None
        if self.filter_maps is not None:
            abs = FilteredAbstraction.from_abstraction(
                self.abstraction, self.filter_maps
            )
        else:
            abs = self.abstraction

        example_input = self.normal_dataset[0][0][None]
        _, example_activations = self.forward_fn(example_input)
        finetuner = AbstractionFinetuner(
            model=abs,
            optimizer=optax.adam(learning_rate=1e-3),
            example_input=example_activations,
            override_variables=self._get_trained_variables(),
            print_tabulate=False,
            normal_weight=self.normal_weight,
            clip=self.clip,
        )

        normal_loader = DataLoader(
            dataset=self.normal_dataset,
            batch_size=self.normal_batch_size,
            collate_fn=abstraction_collate(self.model, self.params),
        )
        new_loader = DataLoader(
            dataset=new_dataset,
            batch_size=self.new_batch_size,
            collate_fn=abstraction_collate(self.model, self.params),
        )

        finetuner.train_model(
            {"normal": normal_loader, "new": new_loader},
            val_loaders={},
            num_epochs=self.num_epochs,
        )

        return self._state_to_dict(finetuner.state)

    def layerwise_scores(self, batch):
        assert self.abstraction is not None
        batch = self._model(batch)
        # TODO: this part should be in helper function in parent class to share code
        if self.abstraction_state is None:
            logger.info("Randomly initializing abstraction.")
            # TODO: should derive this from some other key to avoid
            # hard-coding seed.
            model_rng = jax.random.PRNGKey(0)
            model_rng, init_rng = jax.random.split(model_rng)
            output, activations = batch
            variables = self.abstraction.init(init_rng, activations)
            self.abstraction_state = trainer.InferenceState(
                self.abstraction.apply,
                params=variables["params"],
                batch_stats=variables.get("batch_stats", None),
                rng=model_rng,
            )

        # TODO: should we wrap batch in a Dataset?
        with self.finetune(normal_dataset=self.normal_dataset, new_dataset=batch):
            return compute_losses(
                params=self.abstraction_state.params,
                state=self.abstraction_state,
                batch=batch,
                output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
                return_batch=True,
                layerwise=True,
            )
