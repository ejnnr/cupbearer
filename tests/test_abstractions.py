import pytest
import torch
from cupbearer import models
from cupbearer.detectors.abstraction import AcyclicAbstraction, AutoencoderAbstraction
from torch import nn


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
        AutoencoderAbstraction.get_default,
    ],
)
def test_default_mlp_abstraction_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    expected_dims = [1, 2, 2, 3, 6]
    assert len(abstraction.tau_maps) == len(expected_dims)
    for i in range(len(expected_dims)):
        tau_map = abstraction.tau_maps[f"post_linear_{i}"]
        if i == len(expected_dims) - 1:
            assert isinstance(tau_map, nn.Identity)
        else:
            assert isinstance(tau_map, nn.Linear)
            assert tau_map.in_features == full_dims[i]
            assert tau_map.out_features == expected_dims[i]


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AcyclicAbstraction.get_default],
)
def test_default_mlp_predicted_abstraction_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    expected_dims = [1, 2, 2, 3, 6]
    assert len(abstraction.steps) == len(expected_dims) - 1
    for i in range(1, len(expected_dims)):
        step = abstraction.steps[f"post_linear_{i}"]
        assert isinstance(step, nn.Linear)
        assert step.in_features == expected_dims[i - 1]
        assert step.out_features == expected_dims[i]

    assert "post_linear_0" not in abstraction.steps


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AutoencoderAbstraction.get_default],
)
def test_default_mlp_reconstructed_activation_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    expected_dims = [1, 2, 2, 3, 6]
    assert len(abstraction.decoders) == len(expected_dims)
    for i in range(len(expected_dims) - 1):
        decoder = abstraction.decoders[f"post_linear_{i}"]
        assert isinstance(decoder, nn.Linear)
        assert decoder.in_features == expected_dims[i]
        assert decoder.out_features == full_dims[i]

    assert "post_linear_0" in abstraction.decoders
    assert isinstance(abstraction.decoders[f"post_linear_{i + 1}"], nn.Identity)


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
        AutoencoderAbstraction.get_default,
    ],
)
def test_default_mlp_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    expected_dims = [1, 2, 2, 3, 6]
    inputs = torch.randn(1, 28, 28)
    names = [f"post_linear_{i}" for i in range(len(expected_dims))]
    output, activations = mlp.get_activations(inputs, names=names)
    assert len(activations) == len(expected_dims)
    for name, activation in activations.items():
        assert name.startswith("post_linear_")
        assert activation.ndim == 2
        assert activation.shape[0] == 1
        assert activation.shape[1] == full_dims[int(name[-1])]

    abstractions, _ = abstraction(activations)
    assert abstractions.keys() == _.keys()
    assert abstractions.keys() == activations.keys()
    for k, v in abstractions.items():
        n = int(k[-1])
        assert v.ndim == 2
        assert v.shape[0] == 1
        assert v.shape[1] == expected_dims[n]


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AcyclicAbstraction.get_default],
)
def test_default_mlp_acyclic_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    inputs = torch.randn(1, 28, 28)
    names = [f"post_linear_{i}" for i in range(len(full_dims))]
    output, activations = mlp.get_activations(inputs, names=names)

    abstractions, predicted_abstractions = abstraction(activations)
    for k, v in abstractions.items():
        n = int(k[-1])
        if n == 0:
            assert predicted_abstractions[k] is None
        else:
            assert predicted_abstractions[k].shape == v.shape


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AutoencoderAbstraction.get_default],
)
def test_default_mlp_autoencoder_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5, 6]
    mlp = models.MLP(
        input_shape=(28, 28), hidden_dims=full_dims[:-1], output_dim=full_dims[-1]
    )
    abstraction = get_default_abstraction(mlp, size_reduction=2)
    inputs = torch.randn(1, 28, 28)
    names = [f"post_linear_{i}" for i in range(len(full_dims))]
    output, activations = mlp.get_activations(inputs, names=names)

    abstractions, reconstructed_activations = abstraction(activations)
    for k, v in activations.items():
        assert v.shape == reconstructed_activations[k].shape


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
        AutoencoderAbstraction.get_default,
    ],
)
def test_default_cnn_abstraction_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    expected_dims = [1, 2, 2, 3]
    # There should be one extra tau_map for the final linear layer
    assert len(abstraction.tau_maps) == len(expected_dims) + 1
    for i in range(len(expected_dims)):
        tau_map = abstraction.tau_maps[f"conv_post_conv_{i}"]
        assert isinstance(tau_map, nn.Conv2d)
        assert tau_map.in_channels == full_dims[i]
        assert tau_map.out_channels == expected_dims[i]
    assert isinstance(abstraction.tau_maps["mlp_post_linear_0"], nn.Identity)


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
    ],
)
def test_default_cnn_predicted_abstraction_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    expected_dims = [1, 2, 2, 3]
    # There should be one extra tau_map for the final linear layer
    assert len(abstraction.steps) == len(expected_dims)
    for i in range(1, len(expected_dims)):
        step = abstraction.steps[f"conv_post_conv_{i}"]
        assert isinstance(step, nn.Sequential)
        assert isinstance(step[0], nn.MaxPool2d)
        assert isinstance(step[1], nn.Conv2d)
        assert step[1].in_channels == expected_dims[i - 1]
        assert step[1].out_channels == expected_dims[i]

    assert "post_conv_0" not in abstraction.steps
    # Should be a sequential with pooling + Linear
    assert isinstance(abstraction.steps["mlp_post_linear_0"], nn.Sequential)


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AutoencoderAbstraction.get_default],
)
def test_default_cnn_reconstructed_activations_shapes(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    expected_dims = [1, 2, 2, 3]
    # There should be one extra tau_map for the final linear layer
    assert len(abstraction.decoders) == len(expected_dims) + 1
    for i in range(len(expected_dims)):
        decoder = abstraction.decoders[f"conv_post_conv_{i}"]
        assert isinstance(decoder, nn.Conv2d)
        assert decoder.in_channels == expected_dims[i]
        assert decoder.out_channels == full_dims[i]

    assert "post_conv_0" not in abstraction.decoders
    assert "conv_post_conv_0" in abstraction.decoders
    # Should be a sequential with pooling + Linear
    assert isinstance(abstraction.decoders["mlp_post_linear_0"], nn.Identity)


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
        AutoencoderAbstraction.get_default,
    ],
)
def test_default_cnn_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    expected_dims = [1, 2, 2, 3]
    inputs = torch.randn(1, 1, 28, 28)
    names = [f"conv_post_conv_{i}" for i in range(len(expected_dims))] + [
        "mlp_post_linear_0"
    ]
    output, activations = cnn.get_activations(inputs, names=names)
    # One extra for the MLP activation
    assert len(activations) == len(expected_dims) + 1
    for name, activation in activations.items():
        if name == "mlp_post_linear_0":
            assert activation.ndim == 2
            assert activation.shape[0] == 1
            assert activation.shape[1] == 10
            continue
        assert name.startswith("conv_post_conv_")
        assert activation.ndim == 4
        assert activation.shape[0] == 1
        assert activation.shape[1] == full_dims[int(name[-1])]

    abstractions, _ = abstraction(activations)
    assert abstractions.keys() == _.keys()
    assert abstractions.keys() == activations.keys()
    for k, v in abstractions.items():
        if k == "mlp_post_linear_0":
            assert v.ndim == 2
            assert v.shape[0] == 1
            assert v.shape[1] == 10
            continue
        n = int(k[-1])
        assert v.ndim == 4
        assert v.shape[0] == 1
        assert v.shape[1] == expected_dims[n]


@pytest.mark.parametrize(
    "get_default_abstraction",
    [
        AcyclicAbstraction.get_default,
    ],
)
def test_default_cnn_acyclic_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    inputs = torch.randn(1, 1, 28, 28)
    names = [f"conv_post_conv_{i}" for i in range(len(full_dims))] + [
        "mlp_post_linear_0"
    ]
    output, activations = cnn.get_activations(inputs, names=names)

    abstractions, predicted_abstractions = abstraction(activations)
    assert abstractions.keys() == predicted_abstractions.keys()
    for k, v in abstractions.items():
        if k == "conv_post_conv_0":
            assert predicted_abstractions[k] is None
        else:
            assert predicted_abstractions[k].shape == v.shape
            continue


@pytest.mark.parametrize(
    "get_default_abstraction",
    [AutoencoderAbstraction.get_default],
)
def test_default_cnn_autoencoder_abstraction_forward_pass(get_default_abstraction):
    full_dims = [2, 3, 4, 5]
    cnn = models.CNN(
        input_shape=(1, 28, 28),
        channels=full_dims,
        output_dim=10,
        dense_dims=[],
    )
    abstraction = get_default_abstraction(cnn, size_reduction=2)
    inputs = torch.randn(1, 1, 28, 28)
    names = [f"conv_post_conv_{i}" for i in range(len(full_dims))] + [
        "mlp_post_linear_0"
    ]
    output, activations = cnn.get_activations(inputs, names=names)

    abstractions, reconstructed_activations = abstraction(activations)
    assert reconstructed_activations.keys() == activations.keys()
    for k, v in reconstructed_activations.items():
        assert v.shape == activations[k].shape
        if k == "mlp_post_linear_0":
            assert v.ndim == 2
            assert v.shape[0] == 1
            assert v.shape[1] == cnn.output_dim
            continue
        n = int(k[-1])
        assert v.ndim == 4
        assert v.shape[0] == 1
        assert v.shape[1] == full_dims[n]
