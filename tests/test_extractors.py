import pytest
import torch
from cupbearer import models
from cupbearer.detectors.extractors.activation_extractor import ActivationExtractor
from cupbearer.detectors.extractors.core import FeatureCache


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


class DummyFeatureFn:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.called_with = []

    def __call__(self, inputs):
        self.called_with.append(inputs)
        return {
            name: torch.tensor([hash((name, input)) for input in inputs])
            for name in self.feature_names
        }


def tensor_dicts_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.all(dict1[key] == dict2[key]):
            return False
    return True


@pytest.fixture
def cache():
    return FeatureCache(device="cpu")


FEATURE_NAMES = ["feature_0", "feature_1"]


@pytest.fixture
def dummy_fn():
    return DummyFeatureFn(FEATURE_NAMES)


def test_feature_cache_simple(cache, dummy_fn):
    inputs = ["input1", "input2"]

    expected_features = dummy_fn(inputs)
    assert dummy_fn.called_with == [inputs]

    # Add features to cache
    actual_features = cache.get_features(inputs, FEATURE_NAMES, dummy_fn)
    assert tensor_dicts_equal(actual_features, expected_features)
    assert dummy_fn.called_with == [inputs, inputs]

    # Retrieve features from cache
    retrieved_features = cache.get_features(inputs, FEATURE_NAMES, dummy_fn)
    assert tensor_dicts_equal(retrieved_features, expected_features)
    assert dummy_fn.called_with == [inputs, inputs]


def test_feature_cache_partial_retrieve(cache, dummy_fn):
    inputs = ["input1", "input2"]

    # Add features for only the first input to cache
    features = cache.get_features([inputs[0]], FEATURE_NAMES, dummy_fn)
    assert dummy_fn.called_with == [[inputs[0]]]
    expected_features = dummy_fn([inputs[0]])
    assert dummy_fn.called_with == [[inputs[0]], [inputs[0]]]
    assert tensor_dicts_equal(features, expected_features)

    # Retrieve features from cache, should call feature function for the second input
    retrieved_features = cache.get_features(inputs, FEATURE_NAMES, dummy_fn)
    assert dummy_fn.called_with == [[inputs[0]], [inputs[0]], [inputs[1]]]
    expected_features = dummy_fn(inputs)
    assert tensor_dicts_equal(retrieved_features, expected_features)


def test_feature_cache_mixed_retrieve(cache, dummy_fn):
    inputs = ["input1", "input2"]
    partial_dummy_fn = DummyFeatureFn(FEATURE_NAMES[:1])

    features_1 = cache.get_features([inputs[0]], FEATURE_NAMES[:1], partial_dummy_fn)
    assert partial_dummy_fn.called_with == [[inputs[0]]]
    features_2 = cache.get_features(inputs, FEATURE_NAMES[:1], dummy_fn)
    assert partial_dummy_fn.called_with == [[inputs[0]]]
    assert dummy_fn.called_with == [[inputs[1]]]
    expected_features = partial_dummy_fn([inputs[0]])
    assert tensor_dicts_equal(features_1, expected_features)
    expected_features = partial_dummy_fn(inputs)
    assert tensor_dicts_equal(features_2, expected_features)

    features = cache.get_features([inputs[0]], FEATURE_NAMES, dummy_fn)
    assert dummy_fn.called_with == [[inputs[1]], [inputs[0]]]
    expected_features = dummy_fn([inputs[0]])
    assert tensor_dicts_equal(features, expected_features)
