from pathlib import Path

import pytest
from cupbearer import data, detectors, models, scripts, tasks
from torch.utils.data import DataLoader

pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
def mnist_test():
    return data.MNIST(train=False)


@pytest.fixture(scope="module")
def mnist_train():
    return data.MNIST(train=True)


CACHE_PATH = Path(".cupbearer_cache/performance_tests")
CLASSIFIER_PATH = CACHE_PATH / "backdoored_mnist_mlp"


@pytest.fixture(scope="module")
def model(mnist_train, mnist_test):
    model = models.MLP(input_shape=(28, 28), hidden_dims=[128, 128], output_dim=10)
    try:
        models.load(model=model, path=CLASSIFIER_PATH)
    except FileNotFoundError:
        scripts.train_classifier(
            path=CLASSIFIER_PATH,
            model=model,
            train_loader=DataLoader(
                data.BackdoorDataset(
                    # Poison 5% of the training data
                    original=mnist_train,
                    backdoor=data.CornerPixelBackdoor(p_backdoor=0.05),
                ),
                batch_size=64,
                shuffle=True,
            ),
            num_classes=10,
            max_epochs=3,
        )
        metrics = scripts.eval_classifier(
            data=mnist_test, model=model, path=CLASSIFIER_PATH
        )
        assert metrics[0]["test/acc_epoch"] > 0.9, metrics

    return model


def test_mahalanobis(model, mnist_train, mnist_test, tmp_path):
    detector = detectors.MahalanobisDetector(
        activation_names=[
            "layers.linear_0.output",
            "layers.linear_1.output",
            "layers.linear_2.output",
        ]
    )
    task = tasks.backdoor_detection(
        model, mnist_train, mnist_test, data.CornerPixelBackdoor()
    )

    detector.train(task)
    # Just saving for debugging purposes
    detector.save_weights(tmp_path / "detector")
    metrics, figs = detector.eval(task, layerwise=True, save_path=tmp_path)
    assert metrics["layers.linear_0.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_1.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_2.output"]["AUC_ROC"] > 0.90


def test_supervised_probe(model, mnist_train, mnist_test):
    detector = detectors.SupervisedLinearProbe(
        activation_names=["layers.linear_1.output"]
    )
    task = tasks.backdoor_detection(
        model, mnist_train, mnist_test, data.CornerPixelBackdoor(), trusted_fraction=0.0
    )
    task.untrusted_train_data.return_anomaly_labels = True
    detector.train(task)
    metrics, figs = detector.eval(task, layerwise=True)
    assert metrics["layers.linear_1.output"]["AUC_ROC"] > 0.97


def test_que(tmp_path, model, mnist_train, mnist_test):
    detector = detectors.QuantumEntropyDetector(
        activation_names=[
            "layers.linear_0.output",
            "layers.linear_1.output",
            "layers.linear_2.output",
        ]
    )
    task = tasks.backdoor_detection(
        model, mnist_train, mnist_test, data.CornerPixelBackdoor(), trusted_fraction=0.5
    )

    detector.train(task)
    # Just saving for debugging purposes
    detector.save_weights(tmp_path / "detector")
    metrics, figs = detector.eval(task, layerwise=True, save_path=tmp_path)
    assert metrics["layers.linear_0.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_1.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_2.output"]["AUC_ROC"] > 0.90


def test_vae(model, mnist_train, mnist_test, tmp_path):
    detector = detectors.AbstractionDetector(
        abstraction=detectors.abstraction.VAEAbstraction(
            vaes={
                "layers.linear_0.output": detectors.abstraction.VAE(
                    input_dim=128, latent_dim=32
                ),
                "layers.linear_1.output": detectors.abstraction.VAE(
                    input_dim=128, latent_dim=32
                ),
                "layers.linear_2.output": detectors.abstraction.VAE(
                    input_dim=10, latent_dim=4
                ),
            },
        )
    )
    task = tasks.backdoor_detection(
        model, mnist_train, mnist_test, data.CornerPixelBackdoor()
    )
    detector.train(
        task=task,
        max_epochs=1,
        save_path=tmp_path / "lightning",
    )
    # Just saving for debugging purposes
    detector.save_weights(tmp_path / "detector")
    metrics, figs = detector.eval(task, layerwise=True, save_path=tmp_path)
    assert metrics["layers.linear_0.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_1.output"]["AUC_ROC"] > 0.97
    assert metrics["layers.linear_2.output"]["AUC_ROC"] > 0.90
