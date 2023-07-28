from abstractions.scripts import eval_detector, train_classifier, train_detector
from abstractions.scripts.conf import (
    eval_detector_conf,
    train_classifier_conf,
    train_detector_conf,
)
from abstractions.utils.scripts import save_cfg
from loguru import logger
from simple_parsing import ArgumentGenerationMode, parse


def test_pipeline(tmpdir, capsys):
    ############################
    # Classifier training
    ############################
    logger.info("Running classifier training")
    cfg = parse(
        train_classifier_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'base'} "
        "--train_data mnist --model mlp",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_classifier.main(cfg)

    assert (tmpdir / "base" / "config.yaml").isfile()
    assert (tmpdir / "base" / "model").isdir()
    assert (tmpdir / "base" / "metrics.json").isfile()

    #########################################
    # Abstraction training (backdoor)
    #########################################
    logger.info("Running abstraction training")
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'abstraction'} "
        f"--task backdoor --task.backdoor corner --task.run_path {tmpdir / 'base'} "
        "--detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_detector.main(cfg)
    assert (tmpdir / "abstraction" / "config.yaml").isfile()
    assert (tmpdir / "abstraction" / "detector").isdir()
    assert (tmpdir / "abstraction" / "metrics.json").isfile()

    #########################################
    # Abstraction eval (backdoor)
    #########################################
    logger.info("Running abstraction eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'abstraction'} "
        f"--task backdoor --task.backdoor corner --task.run_path {tmpdir / 'base'} "
        "--detector from_run",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    eval_detector.main(cfg)
    captured = capsys.readouterr()
    assert "Randomly initializing abstraction" not in captured.err
    assert (tmpdir / "abstraction" / "histogram.pdf").isfile()
    assert (tmpdir / "abstraction" / "architecture.png").isfile()
    assert (tmpdir / "abstraction" / "eval.json").isfile()

    ###############################################
    # Mahalanobis training (adversarial examples)
    ###############################################
    logger.info("Running mahalanobis training")
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'mahalanobis'} "
        f"--task adversarial_examples --task.run_path {tmpdir / 'base'} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_detector.main(cfg)
    assert (tmpdir / "mahalanobis" / "config.yaml").isfile()
    assert (tmpdir / "mahalanobis" / "detector").isdir()

    #########################################
    # Mahalanobis eval (adversarial examples)
    #########################################
    logger.info("Running mahalanobis eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'mahalanobis'} "
        f"--task adversarial_examples --task.run_path {tmpdir / 'base'} "
        "--detector from_run",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    eval_detector.main(cfg)
    assert (tmpdir / "mahalanobis" / "histogram.pdf").isfile()
    assert (tmpdir / "mahalanobis" / "architecture.png").isfile()
    assert (tmpdir / "mahalanobis" / "eval.json").isfile()
