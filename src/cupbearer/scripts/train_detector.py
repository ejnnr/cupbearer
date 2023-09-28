import jax
from cupbearer.utils.scripts import run

from . import eval_detector
from .conf import eval_detector_conf
from .conf.train_detector_conf import Config


def main(cfg: Config):
    reference_data = cfg.task.build_train_data()
    model = cfg.task.build_model()
    params = cfg.task.build_params()
    detector = cfg.detector.build(
        model=model,
        params=params,
        save_dir=cfg.dir.path,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    # We want to convert the train dataclass to a dict, but *not* recursively.
    train_kwargs = vars(cfg.detector.train)
    detector.train(reference_data, **train_kwargs)
    if cfg.dir.path is not None:
        detector.save_weights(cfg.dir.path / "detector")
        eval_cfg = eval_detector_conf.Config(
            dir=cfg.dir,
            task=cfg.task,
            seed=cfg.seed,
            debug=cfg.debug,
            debug_with_logging=cfg.debug_with_logging,
        )
        run(eval_detector.main, eval_cfg)


if __name__ == "__main__":
    run(main, Config)
