import pytest
from cupbearer.scripts import eval_detector, train_classifier, train_detector
from cupbearer.scripts.conf import (
    eval_detector_conf,
    train_classifier_conf,
    train_detector_conf,
)
from cupbearer.utils.scripts import run
from loguru import logger
from simple_parsing import ArgumentGenerationMode, parse


@pytest.mark.slow
def test_pipeline(tmp_path, capsys):
    tmp_path.mkdir(exist_ok=True)
    ############################
    # Classifier training
    ############################
    logger.info("Running classifier training")
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'base'} "
        "--train_data backdoor --train_data.original mnist "
        "--train_data.backdoor corner --model mlp",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (tmp_path / "base" / "config.yaml").is_file()
    assert (tmp_path / "base" / "model").is_dir()
    assert (tmp_path / "base" / "metrics.json").is_file()

    #########################################
    # Abstraction training (backdoor)
    #########################################
    logger.info("Running abstraction training")
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'abstraction'} "
        f"--task backdoor --task.backdoor corner --task.path {tmp_path / 'base'} "
        "--detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "abstraction" / "config.yaml").is_file()
    assert (tmp_path / "abstraction" / "detector").is_dir()
    assert (tmp_path / "abstraction" / "metrics.json").is_file()

    captured = capsys.readouterr()
    assert "Randomly initializing abstraction" not in captured.err
    assert (tmp_path / "abstraction" / "histogram.pdf").is_file()
    assert (tmp_path / "abstraction" / "eval.json").is_file()

    #########################################
    # Adversarial abstraction eval (backdoor)
    #########################################
    logger.info("Running adversarial abstraction eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=(
            f"--debug_with_logging --dir.full {tmp_path / 'adversarial_abstraction'} "
            f"--task backdoor --task.backdoor corner "
            f"--task.path {tmp_path / 'base'} "
            "--detector adversarial_abstraction "
            f"--detector.load_path {tmp_path / 'abstraction'} "
            "--save_config true"
        ),
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(eval_detector.main, cfg)
    captured = capsys.readouterr()
    assert "Randomly initializing abstraction" not in captured.err

    for file in {"histogram.pdf", "eval.json", "config.yaml"}:
        assert (tmp_path / "adversarial_abstraction" / file).is_file()

    ###############################################
    # Mahalanobis training (adversarial examples)
    ###############################################
    logger.info("Running mahalanobis training")
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'mahalanobis'} "
        f"--task adversarial_examples --task.path {tmp_path / 'base'} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_detector.main, cfg)
    assert (tmp_path / "mahalanobis" / "config.yaml").is_file()
    assert (tmp_path / "mahalanobis" / "detector").is_dir()
    # Eval outputs:
    assert (tmp_path / "mahalanobis" / "histogram.pdf").is_file()
    assert (tmp_path / "mahalanobis" / "eval.json").is_file()

    ############################
    # WaNet training
    ############################
    logger.info("Running WaNet training")
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'wanet'} "
        "--train_data backdoor --train_data.original gtsrb "
        "--train_data.backdoor wanet --model mlp",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    run(train_classifier.main, cfg)

    assert (tmp_path / "wanet" / "config.yaml").is_file()
    assert (tmp_path / "wanet" / "model").is_dir()
    assert (tmp_path / "wanet" / "metrics.json").is_file()
