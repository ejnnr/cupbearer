from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings

import lightning as L
import torch

from cupbearer import utils
from cupbearer.detectors.activation_based import ActivationBasedDetector
from cupbearer.detectors.extractors import FeatureCache, FeatureExtractor


class FeatureModel(ABC, torch.nn.Module):
    """A model on features that can compute a loss for how well it models them."""

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        pass

    @abstractmethod
    def forward(
        self, inputs: Sequence[Any], features: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Compute a loss for how well this model captures the features.

        Args:
            inputs: The inputs to the main model.
            features: The activations of the main model or features derived
                from them. Shape (batch_size, ...)
            **kwargs: Additional arguments.

        Returns:
            A dictionary of losses, keyed by layer name. Each loss should have a batch
            dimension, corresponding to the batch dimension of the inputs.

            This method may return other outputs if given additional kwargs, but its
            default behavior needs to be returning just this dictionary.

            The dictionary keys must match `self.layer_names`.
        """
        pass


class FeatureModelModule(L.LightningModule):
    def __init__(
        self,
        feature_model: FeatureModel,
        lr: float,
    ):
        super().__init__()

        self.feature_model = feature_model
        self.lr = lr

    def _shared_step(self, batch):
        samples, features = batch
        inputs = utils.inputs_from_batch(samples)
        layer_losses = self.feature_model(inputs, features)
        losses = sum(x for x in layer_losses.values()) / len(layer_losses)
        assert isinstance(losses, torch.Tensor)
        # assert losses.ndim == 1 and len(losses) == len(next(iter(features.values())))
        loss = losses.mean()
        layer_losses = {k: v.mean() for k, v in layer_losses.items()}
        return loss, layer_losses

    def training_step(self, batch, batch_idx):
        loss, layer_losses = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        for k, v in layer_losses.items():
            self.log(f"train/layer_loss/{k}", v)
        return loss

    def configure_optimizers(self):
        # Note we only optimize over the abstraction parameters, the model is frozen
        return torch.optim.Adam(self.feature_model.parameters(), lr=self.lr)


class FeatureModelDetector(ActivationBasedDetector):
    """Anomaly detector based on training some model on activations/features."""

    def __init__(
        self,
        feature_model: FeatureModel,
        feature_extractor: FeatureExtractor | None = None,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
        layer_aggregation: str = "mean",
        cache: FeatureCache | None = None,
    ):
        self.feature_model = feature_model
        super().__init__(
            feature_extractor=feature_extractor,
            activation_names=feature_model.layer_names,
            layer_aggregation=layer_aggregation,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache,
        )

    def _setup_training(self, lr: float):
        self.module = FeatureModelModule(
            self.feature_model,
            lr=lr,
        )

        # Model is not always neccessary, but its abscence should raise a warning
        if self.model is not None:
            warnings.warn(
                "`model` was not set, so standard detector training will not work."
            )
            
            self.model.eval()

            # Pytorch lightning moves the model to the CPU after it's done training.
            # We don't want to expose that behavior to the user, since it's really annoying
            # when not using Lightning.
            self.original_device = next(self.model.parameters()).device

            # HACK: by adding the model as a submodule to the LightningModule, it gets
            # transferred to the same device Lightning uses for everything else
            # (which seems tricky to do manually).
            self.module.model = self.model

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        save_path: Path | str | None = None,
        *,
        lr: float = 1e-3,
        max_epochs: int = 1,
        **trainer_kwargs,
    ):
        if trusted_dataloader is None:
            raise ValueError("Abstraction detector requires trusted training data.")
        self._setup_training(lr)

        if save_path is not None:
            trainer_kwargs["default_root_dir"] = save_path
        else:
            trainer_kwargs["enable_checkpointing"] = False
            trainer_kwargs["logger"] = False

        trainer = L.Trainer(max_epochs=max_epochs, **trainer_kwargs)
        trainer.fit(
            model=self.module,
            train_dataloaders=trusted_dataloader,
        )
        self._teardown_training()

    def _teardown_training(self):
        self.module.to(self.original_device)
        # del self.module

    def _compute_layerwise_scores(self, inputs, features):
        return self.feature_model(inputs, features)

    def _get_trained_variables(self):
        return self.feature_model.state_dict()

    def _set_trained_variables(self, variables):
        self.feature_model.load_state_dict(variables)
