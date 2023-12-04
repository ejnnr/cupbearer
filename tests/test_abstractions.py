from abc import ABC, abstractproperty
from typing import Callable

import torch
from cupbearer import models
from cupbearer.detectors.abstraction import (
    Abstraction,
    AutoencoderAbstraction,
    LocallyConsistentAbstraction,
)
from torch import nn


class ABCTestMLPAbstraction(ABC):
    full_dims = [2, 3, 4, 5, 6]
    model = models.MLP(
        input_shape=(28, 28),
        hidden_dims=full_dims[:-1],
        output_dim=full_dims[-1],
    )
    expected_dims = [1, 2, 2, 3, 6]

    @abstractproperty
    def get_default_abstraction(
        self,
    ) -> Callable[[models.HookedModel, int], Abstraction]:
        pass

    def test_default_abstraction_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        assert len(abstraction.tau_maps) == len(self.expected_dims)
        for i in range(len(self.expected_dims)):
            tau_map = abstraction.tau_maps[f"post_linear_{i}"]
            if i == len(self.expected_dims) - 1:
                assert isinstance(tau_map, nn.Identity)
            else:
                assert isinstance(tau_map, nn.Linear)
                assert tau_map.in_features == self.full_dims[i]
                assert tau_map.out_features == self.expected_dims[i]

    def test_default_abstraction_forward_pass(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 28, 28)
        names = [f"post_linear_{i}" for i in range(len(self.expected_dims))]
        output, activations = self.model.get_activations(inputs, names=names)
        assert len(activations) == len(self.expected_dims)
        for name, activation in activations.items():
            assert name.startswith("post_linear_")
            assert name.split("_")[-1].isdigit()
            assert activation.ndim == 2
            assert activation.shape[0] == 1
            assert activation.shape[1] == self.full_dims[int(name.split("_")[-1])]

        abstractions, _ = abstraction(activations)
        assert abstractions.keys() == _.keys()
        assert abstractions.keys() == activations.keys()
        for k, v in abstractions.items():
            n = int(k[-1])
            assert v.ndim == 2
            assert v.shape[0] == 1
            assert v.shape[1] == self.expected_dims[n]


class ABCTestCNNAbstraction(ABC):
    cnn_dims = [2, 3, 4, 5]
    mlp_dims = [7, 11, 10]
    model = models.CNN(
        input_shape=(1, 28, 28),
        channels=cnn_dims,
        output_dim=mlp_dims[-1],
        dense_dims=mlp_dims[:-1],
    )
    expected_cnn_dims = [1, 2, 2, 3]
    expected_mlp_dims = [4, 6, mlp_dims[-1]]

    @abstractproperty
    def get_default_abstraction(
        self,
    ) -> Callable[[models.HookedModel, int], Abstraction]:
        pass

    def test_default_abstraction_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        # There should be one extra tau_map for the final linear layer
        assert len(abstraction.tau_maps) == (
            len(self.expected_cnn_dims) + len(self.expected_mlp_dims)
        )
        for i in range(len(self.expected_cnn_dims)):
            tau_map = abstraction.tau_maps[f"conv_post_conv_{i}"]
            assert isinstance(tau_map, nn.Conv2d)
            assert tau_map.in_channels == self.cnn_dims[i]
            assert tau_map.out_channels == self.expected_cnn_dims[i]

        for i in range(len(self.mlp_dims) - 1):
            tau_map = abstraction.tau_maps[f"mlp_post_linear_{i}"]
            assert isinstance(tau_map, nn.Linear)
            assert tau_map.in_features == self.mlp_dims[i]
            assert tau_map.out_features == self.expected_mlp_dims[i]
        assert isinstance(abstraction.tau_maps[f"mlp_post_linear_{i + 1}"], nn.Identity)

    def test_default_abstraction_forward_pass(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 1, 28, 28)
        names = [f"conv_post_conv_{i}" for i in range(len(self.expected_cnn_dims))] + [
            f"mlp_post_linear_{i}" for i in range(len(self.expected_mlp_dims))
        ]
        output, activations = self.model.get_activations(inputs, names=names)
        # One extra for the MLP activation
        assert len(activations) == (
            len(self.expected_cnn_dims) + len(self.expected_mlp_dims)
        )
        for name, activation in activations.items():
            if name.startswith("mlp_post_linear_"):
                assert activation.ndim == 2
                assert activation.shape[0] == 1
                assert activation.shape[1] == self.mlp_dims[int(name[-1])]
                continue
            assert name.startswith("conv_post_conv_")
            assert activation.ndim == 4
            assert activation.shape[0] == 1
            assert activation.shape[1] == self.cnn_dims[int(name[-1])]

        abstractions, _ = abstraction(activations)
        assert abstractions.keys() == _.keys()
        assert abstractions.keys() == activations.keys()
        for k, v in abstractions.items():
            if k.startswith("mlp_post_linear_"):
                assert v.ndim == 2
                assert v.shape[0] == 1
                assert v.shape[1] == self.expected_mlp_dims[int(k[-1])]
                continue
            n = int(k[-1])
            assert v.ndim == 4
            assert v.shape[0] == 1
            assert v.shape[1] == self.expected_cnn_dims[n]


class TestMLPLCA(ABCTestMLPAbstraction):
    @property
    def get_default_abstraction(self) -> LocallyConsistentAbstraction:
        return LocallyConsistentAbstraction.get_default

    def test_default_abstraction_step_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        assert len(abstraction.steps) == len(self.expected_dims) - 1
        for i in range(1, len(self.expected_dims)):
            step = abstraction.steps[f"post_linear_{i}"]
            assert isinstance(step, nn.Linear)
            assert step.in_features == self.expected_dims[i - 1]
            assert step.out_features == self.expected_dims[i]

        assert "post_linear_0" not in abstraction.steps

    def test_default_abstraction_forward_pass(self):
        super().test_default_abstraction_forward_pass()
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 28, 28)
        names = [f"post_linear_{i}" for i in range(len(self.full_dims))]
        output, activations = self.model.get_activations(inputs, names=names)

        abstractions, predicted_abstractions = abstraction(activations)
        for k, v in abstractions.items():
            n = int(k[-1])
            if n == 0:
                assert predicted_abstractions[k] is None
            else:
                assert predicted_abstractions[k].shape == v.shape


class TestCNNLCA(ABCTestCNNAbstraction):
    @property
    def get_default_abstraction(self) -> LocallyConsistentAbstraction:
        return LocallyConsistentAbstraction.get_default

    def test_default_abstraction_step_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        # There should be one extra tau_map for the final linear layer
        assert len(abstraction.steps) == (
            -1 + len(self.expected_cnn_dims) + len(self.expected_mlp_dims)
        )
        for i in range(1, len(self.expected_cnn_dims)):
            step = abstraction.steps[f"conv_post_conv_{i}"]
            assert isinstance(step, nn.Sequential)
            assert isinstance(step[0], nn.MaxPool2d)
            assert isinstance(step[1], nn.Conv2d)
            assert step[1].in_channels == self.expected_cnn_dims[i - 1]
            assert step[1].out_channels == self.expected_cnn_dims[i]

        for i in range(1, len(self.expected_mlp_dims)):
            step = abstraction.steps[f"mlp_post_linear_{i}"]
            assert isinstance(step, nn.Linear)
            assert step.in_features == self.expected_mlp_dims[i - 1]
            assert step.out_features == self.expected_mlp_dims[i]

        assert "post_conv_0" not in abstraction.steps
        assert "conv_post_conv_0" not in abstraction.steps
        # Should be a sequential with pooling + Linear
        step = abstraction.steps["mlp_post_linear_0"]
        assert isinstance(step, nn.Sequential)
        assert isinstance(step[0], nn.AdaptiveMaxPool2d)
        assert isinstance(step[1], nn.Flatten)
        assert isinstance(step[2], nn.Linear)
        assert step[2].in_features == self.expected_cnn_dims[-1]
        assert step[2].out_features == self.expected_mlp_dims[0]

    def test_default_abstraction_forward_pass(self):
        super().test_default_abstraction_forward_pass()
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 1, 28, 28)
        names = [f"conv_post_conv_{i}" for i in range(len(self.cnn_dims))] + [
            f"mlp_post_linear_{i}" for i in range(len(self.mlp_dims))
        ]
        output, activations = self.model.get_activations(inputs, names=names)

        abstractions, predicted_abstractions = abstraction(activations)
        assert abstractions.keys() == predicted_abstractions.keys()
        for k, v in abstractions.items():
            if k == "conv_post_conv_0":
                assert predicted_abstractions[k] is None
            else:
                assert predicted_abstractions[k].shape == v.shape
                continue


class TestAutoencoderMLPAbstraction(ABCTestMLPAbstraction):
    @property
    def get_default_abstraction(self) -> AutoencoderAbstraction:
        return AutoencoderAbstraction.get_default

    def test_default_reconstructed_activation_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        assert len(abstraction.decoders) == len(self.expected_dims)
        for i in range(len(self.expected_dims) - 1):
            decoder = abstraction.decoders[f"post_linear_{i}"]
            assert isinstance(decoder, nn.Linear)
            assert decoder.in_features == self.expected_dims[i]
            assert decoder.out_features == self.full_dims[i]

        assert "post_linear_0" in abstraction.decoders
        assert isinstance(abstraction.decoders[f"post_linear_{i + 1}"], nn.Identity)

    def test_default_abstraction_forward_pass(self):
        super().test_default_abstraction_forward_pass()
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 28, 28)
        names = [f"post_linear_{i}" for i in range(len(self.full_dims))]
        output, activations = self.model.get_activations(inputs, names=names)

        abstractions, reconstructed_activations = abstraction(activations)
        for k, v in activations.items():
            assert v.shape == reconstructed_activations[k].shape


class TestCNNAutoencoderAbstraction(ABCTestCNNAbstraction):
    @property
    def get_default_abstraction(self) -> AutoencoderAbstraction:
        return AutoencoderAbstraction.get_default

    def test_default_reconstructed_activations_shapes(self):
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        # There should be one extra tau_map for the final linear layer
        assert len(abstraction.decoders) == (
            len(self.expected_cnn_dims) + len(self.expected_mlp_dims)
        )
        assert "post_conv_0" not in abstraction.decoders
        assert "conv_post_conv_0" in abstraction.decoders
        for i in range(len(self.expected_cnn_dims)):
            decoder = abstraction.decoders[f"conv_post_conv_{i}"]
            assert isinstance(decoder, nn.Conv2d)
            assert decoder.in_channels == self.expected_cnn_dims[i]
            assert decoder.out_channels == self.cnn_dims[i]

        for i in range(len(self.expected_mlp_dims) - 1):
            decoder = abstraction.decoders[f"mlp_post_linear_{i}"]
            assert isinstance(decoder, nn.Linear)
            assert decoder.in_features == self.expected_mlp_dims[i]
            assert decoder.out_features == self.mlp_dims[i]

        assert isinstance(abstraction.decoders[f"mlp_post_linear_{i + 1}"], nn.Identity)

    def test_default_abstraction_forward_pass(self):
        super().test_default_abstraction_forward_pass()
        abstraction = self.get_default_abstraction(self.model, size_reduction=2)
        inputs = torch.randn(1, 1, 28, 28)
        names = [f"conv_post_conv_{i}" for i in range(len(self.cnn_dims))] + [
            f"mlp_post_linear_{i}" for i in range(len(self.mlp_dims))
        ]
        output, activations = self.model.get_activations(inputs, names=names)

        abstractions, reconstructed_activations = abstraction(activations)
        assert reconstructed_activations.keys() == activations.keys()
        for k, v in reconstructed_activations.items():
            assert v.shape == activations[k].shape
            if k.startswith("mlp_post_linear_"):
                assert v.ndim == 2
                assert v.shape[0] == 1
                assert v.shape[1] == self.mlp_dims[int(k[-1])]
                continue
            n = int(k[-1])
            assert v.ndim == 4
            assert v.shape[0] == 1
            assert v.shape[1] == self.cnn_dims[n]
