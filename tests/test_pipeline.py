import pytest
import torch
from cupbearer import data, detectors, models, tasks
from cupbearer.scripts import eval_classifier, train_classifier, train_detector
from torch import nn

# Ignore warnings about num_workers
pytestmark = pytest.mark.filterwarnings(
    "ignore"
    ":The '[a-z]*_dataloader' does not have many workers which may be a bottleneck. "
    "Consider increasing the value of the `num_workers` argument` to "
    "`num_workers=[0-9]*` in the `DataLoader` to improve performance."
    ":UserWarning"
)


@pytest.fixture(scope="module")
def model():
    return models.MLP(input_shape=(1, 28, 28), hidden_dims=[5, 5], output_dim=10)


@pytest.fixture(scope="module")
def abstract_model():
    return models.MLP(input_shape=(1, 28, 28), hidden_dims=[3, 3], output_dim=10)


@pytest.fixture(scope="module")
def mnist():
    # 10 samples will be plenty for all our tests
    return torch.utils.data.Subset(data.MNIST(train=False), range(10))


@pytest.fixture
def backdoor_task(model, mnist):
    return tasks.backdoor_detection(
        model=model,
        train_data=mnist,
        test_data=mnist,
        backdoor=data.CornerPixelBackdoor(),
        # For detectors that need untrusted data
        trusted_fraction=0.5,
    )


@pytest.fixture(scope="module")
def backdoor_classifier_path(model, mnist, module_tmp_path):
    """Trains a backdoored classifier and returns the path to the run directory."""
    dataset = data.BackdoorDataset(original=mnist, backdoor=data.CornerPixelBackdoor())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    train_classifier(
        train_loader=train_loader,
        model=model,
        num_classes=10,
        path=module_tmp_path,
        max_steps=1,
        logger=False,
    )

    assert (module_tmp_path / "checkpoints" / "last.ckpt").is_file()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(model, mnist, backdoor_classifier_path):
    # Test model loading once here; other tests will just use whatever state the model
    # happens to have at that point instead of constantly loading the trained version.
    models.load(model, backdoor_classifier_path)

    eval_classifier(
        data=mnist,
        model=model,
        path=backdoor_classifier_path,
        max_batches=1,
        batch_size=2,
    )

    assert (backdoor_classifier_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_abstraction_corner_backdoor(abstract_model, backdoor_task, tmp_path):
    train_detector(
        task=backdoor_task,
        detector=detectors.AbstractionDetector(
            abstraction=detectors.abstraction.LocallyConsistentAbstraction(
                abstract_model=abstract_model,
                tau_maps={
                    "layers.linear_0.output": nn.Linear(5, 3),
                    "layers.linear_1.output": nn.Linear(5, 3),
                },
            )
        ),
        save_path=tmp_path,
        batch_size=2,
        eval_batch_size=2,
        max_steps=1,
    )
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram_all.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_autoencoder_corner_backdoor(backdoor_task, tmp_path):
    train_detector(
        task=backdoor_task,
        detector=detectors.AbstractionDetector(
            abstraction=detectors.abstraction.AutoencoderAbstraction(
                tau_maps={
                    "layers.linear_0.output": nn.Linear(5, 3),
                    "layers.linear_1.output": nn.Linear(5, 3),
                },
                decoders={
                    "layers.linear_0.output": nn.Linear(3, 5),
                    "layers.linear_1.output": nn.Linear(3, 5),
                },
            )
        ),
        batch_size=2,
        eval_batch_size=2,
        save_path=tmp_path,
        max_steps=1,
    )
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram_all.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_mahalanobis_advex(model, mnist, tmp_path):
    train_detector(
        task=tasks.adversarial_examples(
            model,
            train_data=mnist,
            test_data=mnist,
            cache_path=tmp_path,
            batch_size=2,
            max_examples=2,
            # Success threshold=1.0 means it's fine even if the classifier gets 100%
            # accuracy after the attack---we don't want to error out because of this.
            success_threshold=1.0,
            steps=1,
        ),
        detector=detectors.MahalanobisDetector(
            activation_names=["layers.linear_0.output"]
        ),
        save_path=tmp_path,
        batch_size=2,
        eval_batch_size=2,
        max_steps=1,
    )
    # Note: we don't expect train samples to exist since we have no untrusted train data
    assert not (tmp_path / "adversarial_examples_train.pt").is_file()
    assert not (tmp_path / "adversarial_examples_train.pdf").is_file()
    assert (tmp_path / "adversarial_examples_test.pt").is_file()
    assert (tmp_path / "adversarial_examples_test.pdf").is_file()
    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram_all.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
@pytest.mark.parametrize(
    "detector_type",
    [
        detectors.MahalanobisDetector,
        detectors.SpectralSignatureDetector,
        detectors.QuantumEntropyDetector,
    ],
)
def test_train_statistical_backdoor(tmp_path, backdoor_task, detector_type):
    train_detector(
        task=backdoor_task,
        detector=detector_type(activation_names=["layers.linear_0.output"]),
        batch_size=2,
        eval_batch_size=2,
        save_path=tmp_path,
        max_steps=1,
    )

    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram_all.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_finetuning_detector(backdoor_task, tmp_path):
    train_detector(
        task=backdoor_task,
        detector=detectors.FinetuningAnomalyDetector(),
        save_path=tmp_path,
        num_classes=10,
        batch_size=2,
        eval_batch_size=2,
        max_steps=1,
    )
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram_all.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()
