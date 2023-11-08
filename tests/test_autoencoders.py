import torch
from cupbearer import models
from cupbearer.detectors.autoencoder import get_default_autoencoder
from torch import nn

# TODO test mlp and conv shapes


def test_default_mlp_autoencoder_forward_pass():
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    autoencoder = get_default_autoencoder(mlp, latent_dim=2)
    inputs = torch.randn(1, 28, 28)
    names = [f"post_linear_{i}" for i in range(len(full_dims))]
    output, activations = mlp.get_activations(inputs, names=names)
    assert len(activations) == len(full_dims)
    for name, activation in activations.items():
        assert name.startswith("post_linear_")
        assert activation.ndim == 2
        assert activation.shape[0] == 1
        assert activation.shape[1] == full_dims[int(name[-1])]

    reconstructed_activations = autoencoder(activations)
    assert autoencoder.autoencoders.keys() == activations.keys()
    assert activations.keys() == reconstructed_activations.keys()
    for k, v in reconstructed_activations.items():
        n = int(k[-1])
        assert v.ndim == 2
        assert v.shape[0] == 1
        assert v.shape[1] == full_dims[n]


def test_default_cnn_autoencoder_shapes():
    full_dims = [2, 3, 4, 5]
    n_classes = 17
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=n_classes,
        dense_dims=[],
    )
    autoencoder = get_default_autoencoder(cnn, latent_dim=2)
    # There should be one extra autoencoder for the final linear layer
    assert len(autoencoder.autoencoders) == len(full_dims) + 1
    for i in range(len(full_dims)):
        a = autoencoder.autoencoders[f"conv_post_conv_{i}"]
        assert isinstance(a, nn.Sequential)
        assert a[0].in_channels == full_dims[i]
        assert a[-1].out_channels == full_dims[i]

    mlp_autoencoder = autoencoder.autoencoders["mlp_post_linear_0"]
    assert isinstance(mlp_autoencoder, nn.Sequential)
    assert mlp_autoencoder[0].in_features == n_classes
    assert mlp_autoencoder[-1].out_features == n_classes


def test_default_cnn_autoencoder_forward_pass():
    full_dims = [2, 3, 4, 5]
    n_classes = 17
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=n_classes,
        dense_dims=[],
    )
    autoencoder = get_default_autoencoder(cnn, latent_dim=2)
    inputs = torch.randn(1, 1, 28, 28)
    names = [f"conv_post_conv_{i}" for i in range(len(full_dims))] + [
        "mlp_post_linear_0"
    ]
    output, activations = cnn.get_activations(inputs, names=names)
    # One extra for the MLP activation
    assert len(activations) == len(full_dims) + 1
    for name, activation in activations.items():
        if name == "mlp_post_linear_0":
            assert activation.ndim == 2
            assert activation.shape[0] == 1
            assert activation.shape[1] == n_classes
            continue
        assert name.startswith("conv_post_conv_")
        assert activation.ndim == 4
        assert activation.shape[0] == 1
        assert activation.shape[1] == full_dims[int(name[-1])]

    reconstructed_activations = autoencoder(activations)
    assert autoencoder.autoencoders.keys() == activations.keys()
    assert activations.keys() == reconstructed_activations.keys()
    for k, v in reconstructed_activations.items():
        assert activations[k].shape == v.shape
        if k == "mlp_post_linear_0":
            assert v.ndim == 2
            assert v.shape[0] == 1
            assert v.shape[1] == n_classes
            continue
        n = int(k[-1])
        assert v.ndim == 4
        assert v.shape[0] == 1
        assert v.shape[1] == full_dims[n]
