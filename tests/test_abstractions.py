import pytest
import torch
import transformer_lens.loading_from_pretrained as loading
from cupbearer import utils
from cupbearer.detectors import LocallyConsistentAbstraction
from torch import nn
from transformer_lens import HookedTransformer


@pytest.fixture(scope="module")
def transformer():
    # No strong reason this needs to be a HookedTransformer, just means we don't have
    # to explicitly deal with the tokenizer later.
    # model = HookedTransformer.from_pretrained_no_processing(
    #     "attn-only-1l", device="cpu"
    # )
    official_model_name = loading.get_official_model_name("attn-only-1l")
    cfg = loading.get_pretrained_model_config(
        official_model_name, fold_ln=False, device="cpu"
    )
    model = HookedTransformer(cfg)
    model.init_weights()
    return model


@pytest.fixture
def text():
    return "Hello, world!"


@pytest.fixture
def text_batch():
    return ["Hello, world!", "This is a test."]


def test_lca_identity_abstraction(transformer, text_batch):
    names = [
        name + ".output"
        for name, _ in transformer.named_modules()
        # Filter out pre-softmax attention scores: these have negative infinity entries
        # because of causal masking, which leads to NaN. If we want to ever use these,
        # we'd need a loss function that can deal with this.
        if name and "hook_attn_scores" not in name
    ]
    activations = utils.get_activations(text_batch, model=transformer, names=names)
    # We remove modules that didn't produce any output, don't need tau maps for those:
    names = list(activations.keys())

    tau_maps = {name: nn.Identity() for name in names}
    abstraction = LocallyConsistentAbstraction(
        tau_maps=tau_maps, abstract_model=transformer
    )

    _, abstractions, predicted_abstractions = abstraction(
        text_batch, activations, return_outputs=True
    )

    assert abstractions.keys() == predicted_abstractions.keys() == tau_maps.keys()
    for name in abstractions.keys():
        assert torch.allclose(abstractions[name], predicted_abstractions[name])
        assert torch.all(
            abstraction.loss_fn(name)(
                abstractions[name], predicted_abstractions[name]
            ).abs()
            < 1e-6
        )
