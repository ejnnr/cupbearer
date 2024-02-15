import warnings

from cupbearer.utils.scripts import script

from . import EvalDetectorConfig, eval_detector
from .conf.train_detector_conf import Config


@script
def main(cfg: Config):
    reference_data = cfg.task.build_train_data()
    # reference_data[0] is the first sample, which is (input, ...), so we need another
    # [0] index
    example_input = reference_data[0][0]
    model = cfg.task.build_model(input_shape=example_input.shape)
    detector = cfg.detector.build(model=model, save_dir=cfg.path)

    if cfg.task.normal_weight_when_training < 1.0:
        if not detector.should_train_on_poisoned_data:
            warnings.warn(
                f"Detector of type {type(detector).__name__} is not meant"
                + " to be trained on poisoned samples."
            )
    else:
        if not detector.should_train_on_clean_data:
            warnings.warn(
                f"Detector of type {type(detector).__name__} is not meant"
                + " to be trained without poisoned samples."
            )

    # We want to convert the train dataclass to a dict, but *not* recursively.
    detector.train(
        reference_data,
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
