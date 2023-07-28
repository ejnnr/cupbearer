from abstractions.scripts import train_classifier, train_detector, eval_detector
from abstractions.scripts.conf import (
    train_classifier_conf,
    train_detector_conf,
    eval_detector_conf,
)
from simple_parsing import ArgumentGenerationMode, parse

from abstractions.utils.scripts import save_cfg


def test_pipeline(tmpdir):
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

    cfg = parse(
        train_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'detector'} "
        f"--task backdoor --task.backdoor corner --task.run_path {tmpdir / 'base'} "
        "--detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    save_cfg(cfg)
    train_detector.main(cfg)
    assert (tmpdir / "detector" / "config.yaml").isfile()
    assert (tmpdir / "detector" / "detector").isdir()
    assert (tmpdir / "detector" / "metrics.json").isfile()

    cfg = parse(
        eval_detector_conf.Config,
        args=f"--debug_with_logging --dir.full {tmpdir / 'detector'} "
        f"--task backdoor --task.backdoor corner --task.run_path {tmpdir / 'base'} "
        "--detector abstraction",
        argument_generation_mode=ArgumentGenerationMode.NESTED,
    )
    eval_detector.main(cfg)
    assert (tmpdir / "detector" / "histogram.pdf").isfile()
    assert (tmpdir / "detector" / "architecture.png").isfile()
    assert (tmpdir / "detector" / "eval.json").isfile()
