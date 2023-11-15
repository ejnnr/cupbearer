from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import Abstraction
from cupbearer.detectors.anomaly_detector import (
    ActivationBasedDetector,
)
from cupbearer.models import HookedModel
from cupbearer.utils.optimizers import OptimizerConfig


def compute_losses(
    abstraction_model: Abstraction,
    activations: dict[str, torch.Tensor],
    layerwise=False,
) -> dict[str, torch.Tensor] | torch.Tensor:
    abstractions, predicted_abstractions = abstraction_model(activations)

    # Consistency loss:
    layer_losses: dict[str, torch.Tensor] = {}
    for k in abstractions:
        predicted_abstraction = predicted_abstractions[k]
        if predicted_abstraction is None:
            # We didn't make a prediction for this one
            continue
        actual_abstraction = abstractions[k]
        batch_size = actual_abstraction.shape[0]
        actual_abstraction = actual_abstraction.view(batch_size, -1)
        predicted_abstraction = predicted_abstraction.view(batch_size, -1)
        # Cosine distance can be NaN if one of the inputs is exactly zero
        # which is why we need the eps (which sets cosine distance to 1 in that case).
        # This doesn't happen in realistic scenarios, but in tests with very small
        # hidden dimensions and ReLUs, it's possible.
        predicted_abstraction = F.normalize(predicted_abstraction, dim=1, eps=1e-6)
        actual_abstraction = F.normalize(actual_abstraction, dim=1, eps=1e-6)
        losses = 1 - (predicted_abstraction * actual_abstraction).sum(dim=1)
        layer_losses[k] = losses

    if layerwise:
        return layer_losses

    n = len(layer_losses)
    assert n > 0
    return sum(x for x in layer_losses.values()) / n  # type: ignore


class AbstractionModule(L.LightningModule):
    def __init__(
        self,
        get_activations: Callable[[torch.Tensor], tuple[Any, dict[str, torch.Tensor]]],
        abstraction: Abstraction,
        optim_cfg: OptimizerConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["get_activations", "abstraction"])

        self.get_activations = get_activations
        self.abstraction = abstraction
        self.optim_cfg = optim_cfg

    def _shared_step(self, batch):
        _, activations = self.get_activations(batch)
        losses = compute_losses(self.abstraction, activations)
        assert isinstance(losses, torch.Tensor)
        assert losses.ndim == 1 and len(losses) == len(batch[0])
        loss = losses.mean(0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Note we only optimize over the abstraction parameters, the model is frozen
        return self.optim_cfg.build(self.abstraction.parameters())


class AbstractionDetector(ActivationBasedDetector):
    """Anomaly detector based on an abstraction."""

    def __init__(
        self,
        model: HookedModel,
        abstraction: Abstraction,
        max_batch_size: int = 4096,
        save_path: str | Path | None = None,
    ):
        self.abstraction = abstraction
        names = list(abstraction.tau_maps.keys())
        super().__init__(
            model,
            activation_names=names,
            max_batch_size=max_batch_size,
            save_path=save_path,
        )

    def train(
        self,
        dataset,
        optimizer: OptimizerConfig,
        batch_size: int = 128,
        num_epochs: int = 10,
        validation_datasets: Optional[dict[str, Dataset]] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        # Possibly we should store this as a submodule to save optimizers and continue
        # training later. But as long as we don't actually make use of that,
        # this is easiest.
        module = AbstractionModule(self.get_activations, self.abstraction, optimizer)

        train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        # TODO: implement validation loaders
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=self.save_path,
        #     filename="detector",
        # )

        trainer = L.Trainer(
            max_epochs=num_epochs,
            max_steps=max_steps or -1,
            # callbacks=[checkpoint_callback],
            enable_checkpointing=False,
            logger=None,
            default_root_dir=self.save_path,
        )
        self.model.eval()
        # We don't need gradients for base model parameters:
        required_grad = {}
        for name, param in self.model.named_parameters():
            required_grad[name] = param.requires_grad
            param.requires_grad = False
        # HACK: by adding the model as a submodule to the LightningModule, it gets
        # transferred to the same device Lightning uses for everything else
        # (which seems tricky to do manually).
        module.model = self.model
        trainer.fit(model=module, train_dataloaders=train_loader)

        # Restore original requires_grad values:
        for name, param in self.model.named_parameters():
            param.requires_grad = required_grad[name]

    def layerwise_scores(self, batch):
        _, activations = self.get_activations(batch)
        return compute_losses(self.abstraction, activations, layerwise=True)

    def _get_trained_variables(self, saving: bool = False):
        # TODO: for saving=False we should return optimizer here if we want to make
        # the finetuning API work, I think
        return self.abstraction.state_dict()

    def _set_trained_variables(self, variables):
        self.abstraction.load_state_dict(variables)
