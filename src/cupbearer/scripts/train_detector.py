from cupbearer.utils.scripts import run

from . import eval_detector
from .conf import eval_detector_conf
from .conf.train_detector_conf import Config


def main(cfg: Config):
    reference_data = cfg.task.build_train_data()
    # reference_data[0] is the first sample, which is (input, ...), so we need another
    # [0] index
    example_input = reference_data[0][0]
    model = cfg.task.build_model(input_shape=example_input.shape)
    detector = cfg.detector.build(model=model, save_dir=cfg.path)

    # We want to convert the train dataclass to a dict, but *not* recursively.
    detector.train(
        reference_data,
        num_classes=cfg.task.num_classes,
        train_config=cfg.detector.train,
    )
    if cfg.path:
        detector.save_weights(cfg.path / "detector")
        eval_cfg = eval_detector_conf.Config(
            path=cfg.path,
            task=cfg.task,
            seed=cfg.seed,
        )
        run(eval_detector.main, eval_cfg)


if __name__ == "__main__":
    run(main, Config)
