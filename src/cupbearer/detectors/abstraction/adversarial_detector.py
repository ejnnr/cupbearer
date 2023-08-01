import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import abstraction_collate
from cupbearer.utils import trainer, utils
from cupbearer.utils.trainer import SizedIterable

from .abstraction_detector import OUTPUT_LOSS_FNS, AbstractionDetector, compute_losses


class AbstractionFinetuner(trainer.TrainerModule):
    def __init__(
        self,
        output_loss_fn: str = "kl",
        normal_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.output_loss_fn = OUTPUT_LOSS_FNS[output_loss_fn]
        self.normal_weight = normal_weight
        self.initial_loss = None

    def setup_loss_clipping(self, normal_loader: SizedIterable):
        # We don't want to do this in on_training_start since that's called for
        # every inference forward pass, even though we only need to do this once.
        loss = 0
        for batch in normal_loader:
            new_loss, _ = compute_losses(
                self.state.params,
                self.state,
                batch,
                output_loss_fn=self.output_loss_fn,
            )
            loss += new_loss
        loss /= len(normal_loader)
        self.initial_loss = loss
        logger.info(f"Clipping normal loss to {loss:.4f}")

    def create_functions(self):
        def train_step(state, batch):
            normal_batch, new_batch = batch["normal"], batch["new"]

            def loss_fn(params, batch, normal):
                loss, (_, _, norm) = compute_losses(
                    params, state, batch, output_loss_fn=self.output_loss_fn
                )
                if self.initial_loss is not None and normal:
                    # We don't want to incentivize getting the loss below the initial
                    # loss, just want to make sure it doesn't exceed that
                    loss = jnp.clip(loss, a_min=self.initial_loss)
                return loss, norm

            (normal_loss, normal_norm), normal_grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params, normal_batch, normal=True)

            (new_loss, new_norm), new_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, new_batch, normal=False
            )

            keys = set(normal_grads.keys())
            assert keys == set(new_grads.keys())
            assert keys <= {"tau_maps", "computational_steps"}, keys

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
        new_batch_size: int = 128,
        normal_batch_size: int = 128,
        num_epochs: int = 1,
        normal_weight: float = 0.5,
        clip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.abstraction is not None

        self.normal_dataset = normal_dataset
        self.new_batch_size = new_batch_size
        self.normal_batch_size = normal_batch_size
        self.num_epochs = num_epochs
        self.normal_weight = normal_weight
        self.clip = clip

    def train(self, *args, **kwargs):
        raise NotImplementedError("Use the base AbstractionDetector for pretraining.")

    def eval(
        self,
        normal_dataset: Dataset,
        anomalous_datasets: dict[str, Dataset],
        **kwargs,
    ):
        # We want to run some setup code before the actual eval that we can't run
        # in __init__ because it relies on the datasets.

        # TODO: maybe this should be combined with the normal_loader in the base class?
        self.normal_loader = DataLoader(
            dataset=self.normal_dataset,
            batch_size=self.normal_batch_size,
            collate_fn=abstraction_collate(self.model, self.params),
        )

        example_input = self.normal_dataset[0][0][None]
        _, example_activations = self.forward_fn(example_input)
        self.finetuner = AbstractionFinetuner(
            model=abs,
            optimizer=optax.adam(learning_rate=1e-3),
            example_input=example_activations,
            override_variables=self._get_trained_variables(),
            print_tabulate=False,
            normal_weight=self.normal_weight,
        )
        if self.clip:
            self.finetuner.setup_loss_clipping(self.normal_loader)

        super().eval(normal_dataset, anomalous_datasets, **kwargs)

    def _finetune(
        self,
        new_dataset,
    ) -> dict:
        new_loader = DataLoader(
            dataset=new_dataset,
            batch_size=self.new_batch_size,
            collate_fn=abstraction_collate(self.model, self.params),
        )

        self.finetuner.train_model(
            {"normal": self.normal_loader, "new": new_loader},
            val_loaders={},
            num_epochs=self.num_epochs,
        )

        return self._state_to_dict(self.finetuner.state)

    def layerwise_scores(self, batch):
        batch = self._model(batch)
        self._ensure_abstraction_state(batch)

        # TODO: should we wrap batch in a Dataset?
        with self.finetune(normal_dataset=self.normal_dataset, new_dataset=batch):
            return compute_losses(
                params=self.abstraction_state.params,  # type: ignore
                state=self.abstraction_state,
                batch=batch,
                output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
                return_batch=True,
                layerwise=True,
            )
