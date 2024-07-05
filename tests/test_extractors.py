import pytest
import torch
from cupbearer.detectors.extractors.activation_extractor import ActivationExtractor
from cupbearer.detectors.extractors.core import IdentityExtractor


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Some code needs modules to have at least one parameter to figure out device
        # placement.
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        self.dummy_module = torch.nn.Identity()

    def forward(self, x):
        return self.dummy_module(x)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_batch():
    return torch.randn(10, 3, 28, 28)


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
