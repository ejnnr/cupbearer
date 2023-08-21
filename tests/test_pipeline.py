from cupbearer.scripts import eval_detector, train_classifier, train_detector
from cupbearer.scripts.conf import (
    eval_detector_conf,
    train_classifier_conf,
    train_detector_conf,
)
from cupbearer.utils.scripts import save_cfg
from loguru import logger
from simple_parsing import ArgumentGenerationMode, parse


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
    save_cfg(cfg)
    train_classifier.main(cfg)

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
        f"--task backdoor --task.backdoor corner --task.run_path {tmp_path / 'base'} "
        "--detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_detector.main(cfg)
    assert (tmp_path / "abstraction" / "config.yaml").is_file()
    assert (tmp_path / "abstraction" / "detector").is_dir()
    assert (tmp_path / "abstraction" / "metrics.json").is_file()

    #########################################
    # Adversarial abstraction eval (backdoor)
    #########################################
    logger.info("Running adversarial abstraction eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=(
            f"--debug_with_logging --dir.full {tmp_path / 'adversarial_abstraction'} "
            f"--task backdoor --task.backdoor corner "
            f"--task.run_path {tmp_path / 'base'} "
            "--detector adversarial_abstraction "
            f"--detector.load_path {tmp_path / 'abstraction'} "
            "--save_config true"
        ),
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    eval_detector.main(cfg)
    captured = capsys.readouterr()
    assert "Randomly initializing abstraction" not in captured.err

    for file in {"histogram.pdf", "eval.json", "config.yaml"}:
        assert (tmp_path / "adversarial_abstraction" / file).is_file()

    #########################################
    # Abstraction eval (backdoor)
    #########################################
    logger.info("Running abstraction eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'abstraction'} "
        f"--task backdoor --task.backdoor corner --task.run_path {tmp_path / 'base'} "
        f"--detector from_run --detector.path {tmp_path / 'abstraction'} "
        "--save_config true",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    # It's important we don't overwrite the config file here since it's needed to load
    # the detector correctly!
    save_cfg(cfg, save_config=False)
    eval_detector.main(cfg)
    captured = capsys.readouterr()
    assert "Randomly initializing abstraction" not in captured.err
    assert (tmp_path / "abstraction" / "histogram.pdf").is_file()
    assert (tmp_path / "abstraction" / "eval.json").is_file()

    ###############################################
    # Mahalanobis training (adversarial examples)
    ###############################################
    logger.info("Running mahalanobis training")
    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'mahalanobis'} "
        f"--task adversarial_examples --task.run_path {tmp_path / 'base'} "
        "--detector mahalanobis",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_detector.main(cfg)
    assert (tmp_path / "mahalanobis" / "config.yaml").is_file()
    assert (tmp_path / "mahalanobis" / "detector").is_dir()

    #########################################
    # Mahalanobis eval (adversarial examples)
    #########################################
    logger.info("Running mahalanobis eval")
    cfg = parse(
        eval_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmp_path / 'mahalanobis'} "
        f"--task adversarial_examples --task.run_path {tmp_path / 'base'} "
        f"--detector from_run --detector.path {tmp_path / 'mahalanobis'} "
        "--save_config true",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    eval_detector.main(cfg)
    assert (tmp_path / "mahalanobis" / "histogram.pdf").is_file()
    assert (tmp_path / "mahalanobis" / "eval.json").is_file()
