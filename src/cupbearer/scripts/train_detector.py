from cupbearer.utils.scripts import script

from . import EvalDetectorConfig, eval_detector
from .conf.train_detector_conf import Config


@script
def main(cfg: Config):
    trusted_data = untrusted_data = None

    if cfg.task.allow_trusted:
        trusted_data = cfg.task.trusted_data.build()
        if len(trusted_data) == 0:
            trusted_data = None
    if cfg.task.allow_untrusted:
        untrusted_data = cfg.task.untrusted_data.build()
        if len(untrusted_data) == 0:
            untrusted_data = None

    example_data = trusted_data or untrusted_data
    if example_data is None:
        raise ValueError(
            f"{type(cfg.task).__name__} does not allow trusted nor untrusted data."
        )
    # example_data[0] is the first sample, which is (input, ...), so we need another
    # [0] index
    example_input = example_data[0][0]
    model = cfg.task.build_model(input_shape=example_input.shape)
    detector = cfg.detector.build(model=model, save_dir=cfg.path)

    detector.train(
        trusted_data=trusted_data,
        untrusted_data=untrusted_data,
        num_classes=cfg.task.num_classes,
        train_config=cfg.detector.train,
    )
    if cfg.path:
        detector.save_weights(cfg.path / "detector")
        eval_cfg = EvalDetectorConfig(
            path=cfg.path,
            task=cfg.task,
            seed=cfg.seed,
        )
        eval_detector(eval_cfg)
