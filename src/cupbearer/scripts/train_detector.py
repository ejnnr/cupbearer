import jax
from cupbearer.utils.scripts import run

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


if __name__ == "__main__":
    run(main, Config)
