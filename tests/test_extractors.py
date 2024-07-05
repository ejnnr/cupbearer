import pytest
import torch
from cupbearer import models
from cupbearer.detectors.extractors.activation_extractor import ActivationExtractor
from cupbearer.detectors.extractors.core import IdentityExtractor


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Some code needs modules to have at least one parameter to figure out device
        # placement.
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        self.dummy_module = torch.nn.Identity()
        self.called = False

    def forward(self, x):
        self.called = True
        return self.dummy_module(x)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def mlp():
    return models.MLP(
        input_shape=(3, 8, 8),
        output_dim=10,
        hidden_dims=[4, 6],
    )


@pytest.fixture
def dummy_batch():
    return torch.randn(10, 3, 8, 8)


EXTRACTOR_FACTORIES = [
    IdentityExtractor,
    lambda: ActivationExtractor(names=["dummy_module.output"]),
]


@pytest.mark.parametrize(
    "ExtractorFactory",
    EXTRACTOR_FACTORIES,
)
def test_extractor_basics(ExtractorFactory, dummy_model, dummy_batch):
    extractor = ExtractorFactory()
    assert extractor is not None
    extractor.set_model(dummy_model)
    assert extractor.model is dummy_model
    result = extractor(dummy_batch)
    assert result is not None


def test_activation_extractor_dummy(dummy_model, dummy_batch):
    extractor = ActivationExtractor(names=["dummy_module.output"])
    extractor.set_model(dummy_model)
    acts = extractor(dummy_batch)
    assert dummy_model.called
    assert len(acts) == 1
    assert torch.allclose(acts["dummy_module.output"], dummy_batch)


def test_activation_extractor(mlp, dummy_batch):
    names = (
        [f"layers.linear_{i}.output" for i in range(2)]
        + [f"layers.relu_{i}.output" for i in range(2)]
        + [f"layers.linear_{i}.input" for i in range(2)]
        + [f"layers.relu_{i}.input" for i in range(2)]
    )
    extractor = ActivationExtractor(names=names)
    extractor.set_model(mlp)
    acts = extractor(dummy_batch)
    assert set(acts.keys()) == set(names)
    assert acts["layers.linear_0.output"].shape == (10, 4)
    assert acts["layers.linear_1.output"].shape == (10, 6)
    assert acts["layers.relu_0.output"].shape == (10, 4)
    assert acts["layers.relu_1.output"].shape == (10, 6)
    assert torch.allclose(acts["layers.linear_0.input"], dummy_batch.view(10, -1))
    assert torch.allclose(acts["layers.linear_1.input"], acts["layers.relu_0.output"])
    assert torch.allclose(acts["layers.relu_1.input"], acts["layers.linear_1.output"])
