from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import script


@script
def main(cfg: Config):
    assert cfg.detector is not None  # make type checker happy
    # Init
    train_data = cfg.task.trusted_data
    test_data = cfg.task.test_data
    cfg.detector.set_model(cfg.task.model)

    # Evaluate detector
    cfg.detector.eval(
        train_dataset=train_data,
        test_dataset=test_data,
        pbar=cfg.pbar,
    )
