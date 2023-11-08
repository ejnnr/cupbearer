from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.anomaly_detector import (
    ActivationBasedDetector,
)
from cupbearer.detectors.autoencoder.autoencoder import ActivationAutoencoder
from cupbearer.models import HookedModel
from cupbearer.utils.optimizers import OptimizerConfig


def compute_reconstruction_losses(
    autoencoded_model: ActivationAutoencoder,
    activations: dict[str, torch.Tensor],
    layerwise=False,
) -> dict[str, torch.Tensor] | torch.Tensor:
    reconstructed_activations = autoencoded_model(activations)

    # Reconstruction loss
    layer_losses: dict[str, torch.Tensor] = {}
    for k in activations:
        activation = activations[k]
        reconstructed_activation = reconstructed_activations[k]
        layer_losses[k] = (
            F.kl_div(
                input=reconstructed_activation,
                target=activation,
                reduction="none",
                log_target=True,
            )
            .flatten(start_dim=1)
            .sum(dim=1)
        )

    if layerwise:
        return layer_losses

    return sum(x for x in layer_losses.values()) / len(layer_losses)  # type: ignore


def compute_stability_loss(
    autoencoded_model: ActivationAutoencoder,
    activations: dict[str, torch.Tensor],
    layerwise=False,
):
    raise NotImplementedError
    # TODO compute loss as in paper for within-manifold samples
    # i.e. KL div for output layer when using activations or reconstructed
    # activations. Will need to intervene on activations


class AutoencoderModule(L.LightningModule):
    def __init__(
        self,
        get_activations: Callable[[torch.Tensor], tuple[Any, dict[str, torch.Tensor]]],
        autoencoder: ActivationAutoencoder,
        optim_cfg: OptimizerConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["get_activations", "autoencoder"])

        self.get_activations = get_activations
        self.autoencoder = autoencoder
        self.optim_cfg = optim_cfg

    def _shared_step(self, batch):
        _, activations = self.get_activations(batch)
        losses = compute_reconstruction_losses(self.autoencoder, activations)
        assert isinstance(losses, torch.Tensor)
        assert losses.ndim == 1 and len(losses) == len(batch)
        loss = losses.mean(0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Note we only optimize over the autoencoder parameters, the model is frozen
        return self.optim_cfg.build(self.autoencoder.parameters())


class AutoencoderDetector(ActivationBasedDetector):
    """A generalization of the MagNet detector implemented in:

    Meng, D., & Chen, H. (2017). MagNet: A Two-Pronged Defense against
    Adversarial Examples. Proceedings of the 2017 ACM SIGSAC Conference on
    Computer and Communications Security.
    """

    def __init__(
        self,
        model: HookedModel,
        autoencoder: ActivationAutoencoder,
        max_batch_size: int = 4096,
        temperature: float = 1.0,
        recon_weight: float = 0.5,
        save_path: str | Path | None = None,
    ):
        self.autoencoder = autoencoder
        self.temperature = temperature
        self.recon_weight = recon_weight
        assert 0 <= recon_weight <= 1, "Recon weight has to be between 0 and 1"
        names = list(autoencoder.autoencoders.keys())
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
        debug: bool = False,
        **kwargs,
    ):
        # Possibly we should store this as a submodule to save optimizers and continue
        # training later. But as long as we don't actually make use of that,
        # this is easiest.
        module = AutoencoderModule(self.get_activations, self.autoencoder, optimizer)
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
        output, activations = self.get_activations(batch)
        scores = compute_reconstruction_losses(
            self.autoencoder,
            activations,
            layerwise=True,
        )
        # TODO add stability loss and use recon_weight
        # recon_scores = scores
        # stability_scores = compute_stability_loss(
        #    self.autoencoder,
        #    activations,
        #    layerwise=True,
        # )
        # scores = {
        #    k: (
        #        self.recon_weight * recon_scores[k]
        #        + (1 - self.recon_weight) * stability_scores[k]
        #    ) for k in recon_losses
        # }

        return scores

    def _get_trained_variables(self, saving: bool = False):
        # TODO: for saving=False we should return optimizer here if we want to make
        # the finetuning API work, I think
        return self.autoencoder.state_dict()

    def _set_trained_variables(self, variables):
        self.autoencoder.load_state_dict(variables)
