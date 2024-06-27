from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch

from cupbearer import utils
from cupbearer.detectors.abstraction.abstraction import (
    Abstraction,
    AutoencoderAbstraction,
    LocallyConsistentAbstraction,
)
from cupbearer.detectors.activation_based import (
    ActivationBasedDetector,
    ActivationCache,
)


def compute_losses(
    abstraction: Abstraction,
    inputs,
    activations: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(abstraction, LocallyConsistentAbstraction):
        # LocallyConsistentAbstraction returns (abstractions, predicted_abstractions),
        # where abstractions are the output of tau maps and function as our prediction
        # targets.
        targets, predictions = abstraction(inputs, activations)
    elif isinstance(abstraction, AutoencoderAbstraction):
        # AutoencoderAbstraction returns (abstractions, reconstructed_activations).
        # We don't care about abstractions, our target are the full model's activations.
        _, predictions = abstraction(inputs, activations)
        targets = activations
    else:
        raise ValueError(f"Unsupported abstraction type: {type(abstraction)}")

    layer_losses: dict[str, torch.Tensor] = {}
    assert predictions.keys() == targets.keys()
    for k in predictions.keys():
        if predictions[k] is None:
            # No prediction was made for this layer
            continue
        # prediction = predictions[k].flatten(start_dim=1)
        # target = targets[k].flatten(start_dim=1)

        losses = abstraction.loss_fn(k)(predictions[k], targets[k])
        assert losses.ndim == 1
        layer_losses[k] = losses

    n = len(layer_losses)
    assert n > 0
    return sum(x for x in layer_losses.values()) / n, layer_losses


class AbstractionModule(L.LightningModule):
    def __init__(
        self,
        get_activations: Callable[[Any], dict[str, torch.Tensor]],
        abstraction: Abstraction,
        lr: float,
    ):
        super().__init__()

        self.get_activations = get_activations
        self.abstraction = abstraction
        self.lr = lr

    def _shared_step(self, batch):
        inputs = utils.inputs_from_batch(batch)
        activations = self.get_activations(batch)
        losses, layer_losses = compute_losses(self.abstraction, inputs, activations)
        assert isinstance(losses, torch.Tensor)
        assert losses.ndim == 1 and len(losses) == len(batch[0])
        loss = losses.mean(0)
        layer_losses = {k: v.mean(0) for k, v in layer_losses.items()}
        return loss, layer_losses

    def training_step(self, batch, batch_idx):
        loss, layer_losses = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        for k, v in layer_losses.items():
            self.log(f"train/layer_loss/{k}", v)
        return loss

    def configure_optimizers(self):
        # Note we only optimize over the abstraction parameters, the model is frozen
        return torch.optim.Adam(self.abstraction.parameters(), lr=self.lr)


class AbstractionDetector(ActivationBasedDetector):
    """Anomaly detector based on an abstraction."""

    def __init__(
        self,
        abstraction: Abstraction,
        activation_processing_func: Callable | None = None,
        cache: ActivationCache | None = None,
    ):
        self.abstraction = abstraction
        names = list(abstraction.tau_maps.keys())
        super().__init__(
            activation_names=names,
            activation_processing_func=activation_processing_func,
            cache=cache,
        )

    def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        lr: float = 1e-3,
        batch_size: int = 64,
        **trainer_kwargs,
    ):
        if trusted_data is None:
            raise ValueError("Abstraction detector requires trusted training data.")
        # Possibly we should store this as a submodule to save optimizers and continue
        # training later. But as long as we don't actually make use of that,
        # this is easiest.
        module = AbstractionModule(
            self.get_activations,
            self.abstraction,
            lr=lr,
        )

        train_loader = torch.utils.data.DataLoader(
            trusted_data, batch_size=batch_size, shuffle=True
        )

        # TODO: implement validation data

        self.model.eval()
        # We don't need gradients for base model parameters.
        # TODO: I think this should be unnecessary since `get_activations` uses
        # `torch.no_grad()` anyway.
        required_grad = {}
        for name, param in self.model.named_parameters():
            required_grad[name] = param.requires_grad
            param.requires_grad = False

        # Pytorch lightning moves the model to the CPU after it's done training.
        # We don't want to expose that behavior to the user, since it's really annoying
        # when not using Lightning.
        original_device = next(self.model.parameters()).device

        # HACK: by adding the model as a submodule to the LightningModule, it gets
        # transferred to the same device Lightning uses for everything else
        # (which seems tricky to do manually).
        module.model = self.model

        trainer = L.Trainer(default_root_dir=save_path, **trainer_kwargs)
        trainer.fit(
            model=module,
            train_dataloaders=train_loader,
        )

        module.to(original_device)

        # Restore original requires_grad values:
        for name, param in self.model.named_parameters():
            param.requires_grad = required_grad[name]

    def layerwise_scores(self, batch):
        inputs = utils.inputs_from_batch(batch)
        activations = self.get_activations(batch)
        _, layer_losses = compute_losses(self.abstraction, inputs, activations)
        return layer_losses

    def _get_trained_variables(self, saving: bool = False):
        # TODO: for saving=False we should return optimizer here if we want to make
        # the finetuning API work, I think
        return self.abstraction.state_dict()

    def _set_trained_variables(self, variables):
        self.abstraction.load_state_dict(variables)
