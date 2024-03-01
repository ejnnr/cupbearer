from cupbearer.utils.scripts import script

from . import EvalDetectorConfig, eval_detector
from .conf.train_detector_conf import Config


@script
def main(cfg: Config):
    cfg.detector.set_model(cfg.task.model)

    cfg.detector.train(
        trusted_data=cfg.task.trusted_data,
        untrusted_data=cfg.task.untrusted_train_data,
        num_classes=cfg.num_classes,
        train_config=cfg.train,
    )
    path = cfg.detector.save_path
    if path:
        cfg.detector.save_weights(path / "detector")
        eval_cfg = EvalDetectorConfig(
            detector=cfg.detector,
            task=cfg.task,
            seed=cfg.seed,
        )
        eval_detector(eval_cfg)
