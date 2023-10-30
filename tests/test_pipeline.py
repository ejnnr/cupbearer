import numpy as np
import pytest
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
from cupbearer.utils.scripts import run
from simple_parsing import ArgumentGenerationMode, parse


@pytest.fixture(scope="module")
def classifier_path(module_tmp_path):
    """Trains a classifier and returns the path to the training run directory."""
    module_tmp_path.mkdir(exist_ok=True)
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {module_tmp_path} "
        "--train_data mnist --model mlp",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (module_tmp_path / "config.yaml").is_file()
    assert (module_tmp_path / "model.ckpt").is_file()
    assert (module_tmp_path / "tensorboard").is_dir()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(classifier_path):
    cfg = parse(
        eval_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {classifier_path} "
        "--data mnist --data.train false",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(eval_classifier.main, cfg)

    assert (classifier_path / "eval.json").is_file()


# @pytest.mark.slow
# def test_train_abstraction_corner_backdoor(classifier_path, tmp_path, capsys):
#     cfg = parse(
#         train_detector_conf.Config,
#         args=f"--debug_with_logging --dir.full {tmp_path} "
#         f"--task backdoor --task.backdoor corner --task.path {classifier_path} "
#         "--detector abstraction",
#         argument_generation_mode=ArgumentGenerationMode.NESTED,
#     )
#     run(train_detector.main, cfg)
#     assert (tmp_path / "config.yaml").is_file()
#     assert (tmp_path / "detector.ckpt").is_file()
#     assert (tmp_path / "tensorboard").is_dir()

#     captured = capsys.readouterr()
#     assert "Randomly initializing abstraction" not in captured.err
#     assert (tmp_path / "histogram.pdf").is_file()
#     assert (tmp_path / "eval.json").is_file()


# TODO: add back in adversarial abstractions, will need a trained abstraction as fixture
# probably?
# @pytest.mark.slow
# def test_eval_adversarial_abstraction():
#     cfg = parse(
#         eval_detector_conf.Config,
#         args=(
#             f"--debug_with_logging --dir.full {tmp_path / 'adversarial_abstraction'} "
#             f"--task backdoor --task.backdoor corner "
#             f"--task.path {tmp_path / 'base'} "
#             "--detector adversarial_abstraction "
#             f"--detector.load_path {tmp_path / 'abstraction'} "
#             "--save_config true"
#         ),
#         argument_generation_mode=ArgumentGenerationMode.NESTED,
#     )
#     run(eval_detector.main, cfg)
#     captured = capsys.readouterr()
#     assert "Randomly initializing abstraction" not in captured.err

#     for file in {"histogram.pdf", "eval.json", "config.yaml"}:
#         assert (tmp_path / "adversarial_abstraction" / file).is_file()


# @pytest.mark.slow
# def train_mahalanobis_advex(classifier_path, tmp_path):
#     cfg = parse(
#         train_detector_conf.Config,
#         args=f"--debug_with_logging --dir.full {tmp_path} "
#         f"--task adversarial_examples --task.path {classifier_path} "
#         "--detector mahalanobis",
#         argument_generation_mode=ArgumentGenerationMode.NESTED,
#     )
#     run(train_detector.main, cfg)
#     assert (tmp_path / "config.yaml").is_file()
#     assert (tmp_path / "detector.ckpt").is_file()
#     # Eval outputs:
#     assert (tmp_path / "histogram.pdf").is_file()
#     assert (tmp_path / "eval.json").is_file()


@pytest.mark.slow
def train_mahalanobis_backdoor(classifier_path, tmp_path):
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path} "
        f"--task backdoor --task.backdoor corner --task.path {classifier_path} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "detector.ckpt").is_file()
    # Eval outputs:
<<<<<<< HEAD
    assert (tmp_path / "mahalanobis" / "histogram.pdf").is_file()
    assert (tmp_path / "mahalanobis" / "eval.json").is_file()


@pytest.mark.slow
def test_wanet(tmp_path, capsys):
    tmp_path.mkdir(exist_ok=True)
    ############################
    # WaNet
    ############################
    logger.info("Running WaNet training")
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'wanet'} "
        "--train_data backdoor --train_data.original gtsrb "
        "--train_data.backdoor wanet --model mlp "
        "--val_data.backdoor backdoor --val_data.backdoor.original gtsrb "
        "--val_data.backdoor.backdoor wanet",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (tmp_path / "wanet" / "config.yaml").is_file()
    assert (tmp_path / "wanet" / "model").is_dir()
    assert (tmp_path / "wanet" / "metrics.json").is_file()
    # Check that NoData is handled correctly
    for name, data_cfg in cfg.val_data.items():
        if name == "backdoor":
            assert np.allclose(
                data_cfg.backdoor.warping_field,
                cfg.train_data.backdoor.warping_field,
            )
        else:
            with pytest.raises(NotImplementedError):
                data_cfg.build()

    # Check that from_run can load WanetBackdoor properly
    train_detector_cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'wanet-mahalanobis'} "
        f"--task backdoor --task.backdoor wanet --task.path {tmp_path / 'wanet'} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, train_detector_cfg)
    assert np.allclose(
        train_detector_cfg.task.backdoor.warping_field,
        cfg.train_data.backdoor.warping_field,
    )
=======
    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()
>>>>>>> fc10822 (WIP for pytorch transition)
