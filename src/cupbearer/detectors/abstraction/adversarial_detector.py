import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import abstraction_collate
from cupbearer.utils import utils
from cupbearer.utils.trainer import SizedIterable, TrainState

from .abstraction_detector import OUTPUT_LOSS_FNS, AbstractionDetector, compute_losses


# itertools.cycle would load all the Dataloader's batches into memory, this version
# recreates the Dataloader instead when it runs out.
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class AdversarialAbstractionDetector(AbstractionDetector):
    def __init__(
        self,
        num_at_once: int = 1,
        num_ref_samples: int = 128,
        num_steps: int = 1,
        normal_weight: float = 0.5,
        clip: bool = True,
        **kwargs,
    ):
        # max_batch_size is used by eval to create its dataloaders.
        # Calling it max_batch_size would be pretty misleading for this class
        # since the batch size actually has a crucial effect on the results.
        kwargs["max_batch_size"] = num_at_once
        super().__init__(**kwargs)
        assert self.abstraction is not None

        self.num_at_once = num_at_once
        self.num_ref_samples = num_ref_samples
        self.num_steps = num_steps
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
        # TODO: consider adding hooks to the base class for this kind of thing
        self._setup_finetuning(normal_dataset)

        super().eval(normal_dataset, anomalous_datasets, **kwargs)

    def _setup_finetuning(self, reference_dataset: Dataset):
        # TODO: maybe this should be combined with the normal_loader in the base class?
        self.reference_loader = DataLoader(
            dataset=reference_dataset,
            batch_size=self.num_ref_samples,
            collate_fn=abstraction_collate(self.model, self.params),
            drop_last=True,
        )
        # Infinite version of the dataloader:
        self.ref_batches = cycle(self.reference_loader)

        # In case we don't have an abstraction state, we randomly initialize it.
        # (This happens if the user just doesn't load a pretrained detector.)
        example_input = reference_dataset[0][0][None]
        example_input = self._model(example_input)
        self._ensure_abstraction_state(example_input)

        assert self.abstraction_state is not None  # make type checkers happy
        # self.abstraction_state is now probably an InferenceState,
        # we need to add the optimizer.
        if not isinstance(self.abstraction_state, TrainState):
            # TODO: optimizer should be configurable.
            tx = optax.adam(learning_rate=1e-3)
            self.abstraction_state = TrainState.from_inference_state(
                self.abstraction_state, tx=tx
            )

        assert isinstance(self.abstraction_state, TrainState)

        if self.clip:
            self._setup_loss_clipping(self.reference_loader)

    def _setup_loss_clipping(self, reference_loader: SizedIterable):
        # We don't want to do this in on_training_start since that's called for
        # every inference forward pass, even though we only need to do this once.
        loss = 0
        for batch in reference_loader:
            new_loss, _ = compute_losses(
                self.abstraction_state.params,
                self.abstraction_state,
                batch,
                output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
            )
            loss += new_loss
        loss /= len(reference_loader)
        self.initial_loss = loss
        logger.info(f"Clipping normal loss to {loss:.4f}")

    def _finetune(
        self,
        new_batch,
    ) -> dict:
        assert isinstance(self.abstraction_state, TrainState), self.abstraction_state

        def loss_fn(params, batch, normal):
            loss, (_, _, norm) = compute_losses(
                params,
                self.abstraction_state,
                batch,
                output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
            )
            if self.initial_loss is not None and normal:
                # We don't want to incentivize getting the loss below the initial
                # loss, just want to make sure it doesn't exceed that
                loss = jnp.clip(loss, a_min=self.initial_loss)
            return loss, norm

        for _ in range(self.num_steps):
            ref_batch = next(self.ref_batches)

            (normal_loss, normal_norm), normal_grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(self.abstraction_state.params, ref_batch, normal=True)

            (new_loss, new_norm), new_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                self.abstraction_state.params, new_batch, normal=False
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

            self.abstraction_state = self.abstraction_state.apply_gradients(grads=grads)
            metrics = {
                "normal_loss": normal_loss,
                "new_loss": new_loss,
                "normal_norm": normal_norm,
                "new_norm": new_norm,
            }
            print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        return self._get_trained_variables()

    def layerwise_scores(self, batch):
        batch = self._model(batch)

        with self.finetune(new_batch=batch):
            return compute_losses(
                params=self.abstraction_state.params,  # type: ignore
                state=self.abstraction_state,
                batch=batch,
                output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
                return_batch=True,
                layerwise=True,
            )
