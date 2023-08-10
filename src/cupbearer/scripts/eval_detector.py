import jax
from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import run


def main(cfg: Config):
    reference_data = cfg.task.build_train_data()
    anomalous_data = cfg.task.build_test_data()
    model = cfg.task.build_model()
    params = cfg.task.build_params()
    detector = cfg.detector.build(
        model=model,
        params=params,
        rng=jax.random.PRNGKey(cfg.seed),
        save_dir=cfg.dir.path,
    )

    detector.eval(
        normal_dataset=reference_data,
        anomalous_datasets={"anomalous": anomalous_data},
    )


if __name__ == "__main__":
    run(main, Config)
