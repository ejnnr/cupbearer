import pytest
import torch
from cupbearer import data, detectors, models, tasks
from cupbearer.scripts import (
    eval_classifier,
    train_classifier,
    train_detector,
)
from cupbearer.scripts.conf import (
    eval_classifier_conf,
    train_classifier_conf,
    train_detector_conf,
)
from cupbearer.utils.train import DebugTrainConfig

# Ignore warnings about num_workers
pytestmark = pytest.mark.filterwarnings(
    "ignore"
    ":The '[a-z]*_dataloader' does not have many workers which may be a bottleneck. "
    "Consider increasing the value of the `num_workers` argument` to "
    "`num_workers=[0-9]*` in the `DataLoader` to improve performance."
    ":UserWarning"
)


@pytest.fixture(scope="module")
def backdoor_classifier_path(module_tmp_path):
    """Trains a backdoored classifier and returns the path to the run directory."""
    cfg = train_classifier_conf.DebugConfig(
        train_data=data.BackdoorData(
            original=data.MNIST(), backdoor=data.CornerPixelBackdoor()
        ),
        model=models.DebugMLPConfig(),
        path=module_tmp_path,
    )
    train_classifier(cfg)

    assert (module_tmp_path / "config.yaml").is_file()
    assert (module_tmp_path / "checkpoints" / "last.ckpt").is_file()
    assert (module_tmp_path / "tensorboard").is_dir()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(backdoor_classifier_path):
    cfg = eval_classifier_conf.DebugConfig(
        path=backdoor_classifier_path, data=data.MNIST(train=False)
    )

    eval_classifier(cfg)

    assert (backdoor_classifier_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_abstraction_corner_backdoor(backdoor_classifier_path, tmp_path):
    cfg = train_detector_conf.Config(
        task=tasks.BackdoorDetection(path=backdoor_classifier_path),
        detector=detectors.AbstractionDetectorConfig(train=DebugTrainConfig()),
        path=tmp_path,
    )
    train_detector(cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()

    assert (tmp_path / "tensorboard").is_dir()


@pytest.mark.slow
def test_train_autoencoder_corner_backdoor(backdoor_classifier_path, tmp_path):
    cfg = train_detector_conf.Config(
        task=tasks.BackdoorDetection(path=backdoor_classifier_path),
        detector=detectors.AbstractionDetectorConfig(
            train=DebugTrainConfig(),
            abstraction=detectors.abstraction.AutoencoderAbstractionConfig(),
        ),
        path=tmp_path,
    )
    train_detector(cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()

    assert (tmp_path / "tensorboard").is_dir()


@pytest.mark.slow
def test_train_mahalanobis_advex(backdoor_classifier_path, tmp_path):
    # This test doesn't need a backdoored classifier, but we already have one
    # and it doesn't hurt, so reusing it makes execution faster.
    cfg = train_detector_conf.Config(
        task=tasks.adversarial_examples.DebugAdversarialExampleTask(
            path=backdoor_classifier_path
        ),
        detector=detectors.DebugMahalanobisConfig(),
        path=tmp_path,
    )
    train_detector(cfg)
    assert (backdoor_classifier_path / "adv_examples_train.pt").is_file()
    assert (backdoor_classifier_path / "adv_examples.pdf").is_file()
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
@pytest.mark.parametrize(
    "detector_type",
    [
        detectors.DebugMahalanobisConfig,
        detectors.DebugSpectralSignatureConfig,
        detectors.DebugQuantumEntropyConfig,
    ],
)
def test_train_statistical_backdoor(backdoor_classifier_path, tmp_path, detector_type):
    cfg = train_detector_conf.Config(
        task=tasks.backdoor_detection.DebugBackdoorDetection(
            # Need some untrusted data for SpectralSignatureConfig
            path=backdoor_classifier_path,
            trusted_fraction=0.5,
        ),
        detector=detector_type(),
        path=tmp_path,
    )

    train_detector(cfg)

    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()
    # Eval outputs:
    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def test_finetuning_detector(backdoor_classifier_path, tmp_path):
    cfg = train_detector_conf.Config(
        task=tasks.BackdoorDetection(path=backdoor_classifier_path),
        detector=detectors.finetuning.FinetuningConfig(train=DebugTrainConfig()),
        path=tmp_path,
    )
    train_detector(cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()

    assert (tmp_path / "tensorboard").is_dir()


@pytest.mark.slow
def test_wanet(tmp_path):
    cfg = train_classifier_conf.DebugConfig(
        train_data=data.BackdoorData(
            original=data.GTSRB(), backdoor=data.WanetBackdoor()
        ),
        model=models.DebugMLPConfig(),
        path=tmp_path / "wanet",
        val_data={
            "backdoor": data.BackdoorData(
                original=data.GTSRB(), backdoor=data.WanetBackdoor()
            )
        },
        train_config=DebugTrainConfig(num_workers=1),
    )
    train_classifier(cfg)

    assert (tmp_path / "wanet" / "config.yaml").is_file()
    assert (tmp_path / "wanet" / "checkpoints" / "last.ckpt").is_file()
    assert (tmp_path / "wanet" / "tensorboard").is_dir()

    # Checks mostly to make the type checker happy for the allclose assert
    assert isinstance(cfg.val_data["backdoor"], data.BackdoorData)
    assert isinstance(cfg.val_data["backdoor"].backdoor, data.WanetBackdoor)
    assert isinstance(cfg.train_data, data.BackdoorData)
    assert isinstance(cfg.train_data.backdoor, data.WanetBackdoor)
    assert torch.allclose(
        cfg.val_data["backdoor"].backdoor.control_grid,
        cfg.train_data.backdoor.control_grid,
    )

    # Check that from_run can load WanetBackdoor properly
    train_detector_cfg = train_detector_conf.Config(
        task=tasks.backdoor_detection.DebugBackdoorDetection(path=tmp_path / "wanet"),
        detector=detectors.DebugMahalanobisConfig(),
        path=tmp_path / "wanet-mahalanobis",
    )
    train_detector(train_detector_cfg)
    assert isinstance(train_detector_cfg.task, tasks.BackdoorDetection)
    assert isinstance(train_detector_cfg.task._backdoor, data.WanetBackdoor)
    assert torch.allclose(
        train_detector_cfg.task._backdoor.control_grid,
        cfg.train_data.backdoor.control_grid,
    )
