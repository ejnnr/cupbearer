import torch
from torch import nn

from cupbearer import utils

from .feature_model_detector import FeatureModel, FeatureModelDetector


class VAE(nn.Module):
    """Simple VAE with MLP encoder and decoder.

    Adapted from PyTorch VAE (Apache-2.0) but with a different architecture.
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

    Original copyright notice:
                             Apache License
                       Version 2.0, January 2004
                    http://www.apache.org/licenses/
                    Copyright Anand Krishnamoorthy Subramanian 2020
                               anandkrish894@gmail.com
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, 2 * self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, input_dim),
        )

    def encode(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        assert input.ndim == 2
        result = self.encoder(input)
        assert result.ndim == 2
        assert result.shape[1] == 2 * self.latent_dim

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = result[:, : self.latent_dim]
        log_var = result[:, self.latent_dim :]

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self, input: torch.Tensor, noise: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(input)
        if noise:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return self.decode(z), mu, log_var

    def loss_function(
        self, reconstruction, input, mu, log_var, kld_weight=1.0, reduce: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) =
        \\log \frac{1}{\\sigma} + \frac{\\sigma^2 + \\mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = nn.functional.mse_loss(reconstruction, input, reduction="none")
        # Reduce over all but first dimension
        recons_loss = recons_loss.view(recons_loss.shape[0], -1).mean(dim=1)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)

        if reduce:
            recons_loss = recons_loss.mean()
            kld_loss = kld_loss.mean()

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }


class VAEFeatureModel(FeatureModel):
    def __init__(self, vaes: dict[str, VAE], kld_weight: float = 1.0):
        super().__init__()
        self.vaes = utils.ModuleDict(vaes)
        self.kld_weight = kld_weight

    @property
    def layer_names(self):
        return list(self.vaes.keys())

    def forward(
        self, inputs, features: dict[str, torch.Tensor], return_outputs: bool = False
    ) -> dict[str, torch.Tensor]:
        vae_outputs = {
            name: vae(features[name], noise=False) for name, vae in self.vaes.items()
        }
        # VAE outputs are (reconstruction, mu, log_var)
        reconstructions = {
            name: vae_output[0] for name, vae_output in vae_outputs.items()
        }
        mus = {name: vae_output[1] for name, vae_output in vae_outputs.items()}
        log_vars = {name: vae_output[2] for name, vae_output in vae_outputs.items()}

        losses = {
            name: self.vaes[name].loss_function(
                reconstructions[name],
                features[name],
                mus[name],
                log_vars[name],
                kld_weight=self.kld_weight,
                reduce=False,
            )
            for name in self.layer_names
        }
        if return_outputs:
            return losses, reconstructions, mus, log_vars

        return {name: loss["loss"] for name, loss in losses.items()}


class VAEDetector(FeatureModelDetector):
    def __init__(self, vaes: dict[str, VAE], kld_weight: float = 1.0, **kwargs):
        super().__init__(VAEFeatureModel(vaes, kld_weight), **kwargs)
