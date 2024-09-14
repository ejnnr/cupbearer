from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from cupbearer import utils

from .feature_model_detector import FeatureModel


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


class Abstraction(FeatureModel, ABC):
    tau_maps: utils.ModuleDict

    @abstractmethod
    def loss_fn(
        self, name: str
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the loss function for the given activation.

        The loss function should take a prediction and a target,
        each of shape (batch, dim), and return a tensor of shape (batch,).
        """
        pass

    def forward(self, inputs, activations: dict[str, torch.Tensor]):
        """Get the abstractions for the given inputs and activations.

        Args:
            inputs: The inputs to the model.
            activations: A dictionary mapping activation names to the activations of the
                full model at that name.
        """
        raise NotImplementedError


def l2_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert input.shape == target.shape
    input = input.reshape(input.size(0), -1)
    target = target.reshape(target.size(0), -1)
    # Not totally clear whether mean or sum makes more sense here.
    return F.mse_loss(input, target, reduction="none").mean(dim=1)


def cosine_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert input.shape == target.shape
    input = input.reshape(input.size(0), -1)
    target = target.reshape(target.size(0), -1)
    # Cosine distance can be NaN if one of the inputs is exactly zero
    # which is why we need the eps (which sets cosine distance to 1 in that case).
    # This doesn't happen in realistic scenarios, but in tests with very small
    # hidden dimensions and ReLUs, it's possible.
    return 1 - F.cosine_similarity(input, target, eps=1e-6)


def kl_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Expects logits for input and normalized logprobs for target
    assert input.shape == target.shape
    assert input.ndim >= 2
    batch_size = input.size(0)
    class_size = input.size(-1)
    input = input.reshape(-1, class_size)
    target = target.reshape(-1, class_size)

    input = F.log_softmax(input, dim=1)
    loss = F.kl_div(input, target, reduction="none", log_target=True).sum(dim=-1)
    return loss.reshape(batch_size, -1).mean(dim=1)


def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Expects logits for input and probabilities for target.
    # (Note that unlike for KL divergence, it's fine for numerical stability
    # if target probabilities are very small.) It's sad from a user perspective
    # though that these have different conventions.
    assert input.shape == target.shape
    assert input.ndim >= 2
    batch_size = input.size(0)
    class_size = input.size(-1)
    input = input.view(-1, class_size)
    target = target.view(-1, class_size)

    # Note that even with reduction="none", cross_entropy does reduce over the class
    # axis, just not the batch axis. This is already what we want. For kl_div above,
    # we had to manually sum over the class axis.
    loss = F.cross_entropy(input, target, reduction="none")
    return loss.view(batch_size, -1).mean(dim=1)


class LocallyConsistentAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],
        abstract_model: nn.Module,
        loss_fns: dict[str, Callable] | None = None,
        loss_weights: dict[str, float] | None = None,
        activation_processing_func: Callable | None = None,
        global_consistency: bool = False,
    ):
        super().__init__()
        self.tau_maps = utils.ModuleDict(tau_maps)
        self.abstract_model = abstract_model
        self.loss_fns = loss_fns or {}
        self.loss_weights = loss_weights or {}
        # activation_processing_func is used on the abstract predicted activations
        # before they are compared to the tau_map output. This can in principle be
        # distinct from the activation_processing_func set in the detector
        # (which applies to the full model's activations), but it works the same way.
        self.activation_processing_func = activation_processing_func
        self.global_consistency = global_consistency

    def loss_fn(
        self, name: str
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        fn = self.loss_fns.get(name, l2_loss)
        if name in self.loss_weights:
            return lambda input, target: fn(input, target) * self.loss_weights[name]
        return fn

    def forward(
        self, inputs, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        # TODO: given that this is called in each forward pass, it might be worth
        # switching to https://github.com/metaopt/optree for faster tree maps
        assert activations.keys() == self.tau_maps.keys()
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        predicted_abstractions = {}

        # This is similar to `get_activations`, except we also intervene in the forward
        # pass. Maybe we could unify them?
        def make_output_hook(name):
            def hook(module, input, output):
                # First, we cache the value that the abstract model predicted here:
                predicted_abstractions[name] = output

                # Then, we replace it with the "true" value (i.e. the one from the
                # tau map) for the rest of the forward pass. (This is what makes this
                # a *local* abstraction.)
                if not self.global_consistency:
                    return abstractions[name]

            return hook

        def make_input_hook(name):
            # This will be used as a *pre*-hook, so doesn't get an output argumet.
            # Important since we want to modify the *input* to the module here,
            # so can't use post-hooks.
            def hook(module, input):
                if isinstance(input, torch.Tensor):
                    predicted_abstractions[name] = input
                elif isinstance(input[0], torch.Tensor):
                    predicted_abstractions[name] = input[0]
                else:
                    raise ValueError(
                        "Expected input to be a tensor or tuple with tensor as "
                        f"first element, got {type(input)}"
                    )

                if not self.global_consistency:
                    return abstractions[name]

            return hook

        hooks = []

        try:
            for name, module in self.abstract_model.named_modules():
                if name + ".input" in self.tau_maps:
                    hooks.append(
                        module.register_forward_pre_hook(
                            make_input_hook(name + ".input")
                        )
                    )
                if name + ".output" in self.tau_maps:
                    hooks.append(
                        module.register_forward_hook(make_output_hook(name + ".output"))
                    )

            self.abstract_model(inputs)

        finally:
            for hook in hooks:
                hook.remove()

        assert abstractions.keys() == predicted_abstractions.keys()

        if self.activation_processing_func is not None:
            predicted_abstractions = {
                k: self.activation_processing_func(v, inputs, k)
                for k, v in predicted_abstractions.items()
            }

        return abstractions, predicted_abstractions


class AutoencoderAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],  # encoders
        decoders: dict[str, nn.Module],  # decoders
    ):
        super().__init__()
        assert tau_maps.keys() == decoders.keys()
        self.tau_maps = utils.ModuleDict(tau_maps)
        self.decoders = utils.ModuleDict(decoders)

    def loss_fn(self, name: str) -> Callable:
        return l2_loss

    def forward(
        self, inputs, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        reconstructed_activations: dict[str, torch.Tensor | None] = {
            name: self.decoders[name](abstraction)
            for name, abstraction in abstractions.items()
        }

        return abstractions, reconstructed_activations
