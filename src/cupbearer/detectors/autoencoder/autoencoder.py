import torch
from torch import nn

from cupbearer import models


class ActivationAutoencoder(nn.Module):
    def __init__(
        self,
        autoencoders: dict[str, nn.Module],
    ):
        super().__init__()
        self.autoencoders = nn.ModuleDict(autoencoders)
        self.names = tuple(autoencoders.keys())

    def forward(self, activations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reconstructed_activations: dict[str, torch.Tensor] = {
            name: self.autoencoders[name](activation)
            for name, activation in activations.items()
        }

        return reconstructed_activations


def get_default_autoencoder(
    model: models.HookedModel, latent_dim: int
) -> ActivationAutoencoder:
    def get_mlp_autoencoders(mlp: models.MLP) -> dict[str, nn.Module]:
        full_dims = mlp.hidden_dims + [mlp.output_dim]
        return {  # TODO choose architecture
            f"post_linear_{i}": nn.Sequential(
                nn.Linear(in_features, latent_dim),
                nn.Linear(latent_dim, in_features),
            )
            for i, in_features in enumerate(full_dims)
        }

    autoencoders = {}
    if isinstance(model, models.MLP):
        autoencoders.update(get_mlp_autoencoders(model))

    elif isinstance(model, models.CNN):
        # Conv autoencoder architecture is a variant of one in the MagNet paper
        conv_autoencoders = {
            f"post_conv_{i}": nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=3, padding="same"),
                nn.ReLU(),  # original uses sigmoid
                nn.Conv2d(3, 3, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(3, 1, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(1, in_channels, kernel_size=3, padding="same"),
            )
            for i, in_channels in enumerate(model.channels)
        }
        mlp_autoencoders = get_mlp_autoencoders(model.mlp)
        for name in conv_autoencoders:
            assert f"conv_{name}" not in autoencoders
            autoencoders[f"conv_{name}"] = conv_autoencoders[name]
        for name in mlp_autoencoders:
            assert f"mlp_{name}" not in autoencoders
            autoencoders[f"mlp_{name}"] = mlp_autoencoders[name]
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    return ActivationAutoencoder(autoencoders)
