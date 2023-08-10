import jax
from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import run


def main(cfg: Config):
    train_data = cfg.task.build_train_data()
    test_data = cfg.task.build_test_data()
    model = cfg.task.build_model()
    params = cfg.task.build_params()
    detector = cfg.detector.build(
        model=model,
        params=params,
        rng=jax.random.PRNGKey(cfg.seed),
        save_dir=cfg.dir.path,
    )

    detector.eval(
        train_dataset=train_data,
        test_dataset=test_data,
        pbar=cfg.pbar,
    )


if __name__ == "__main__":
    run(main, Config)
