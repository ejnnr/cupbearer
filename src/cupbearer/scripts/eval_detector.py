import jax
from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import run
from torch.utils.data import Subset


def main(cfg: Config):
    reference_data = cfg.task.build_reference_data()
    anomalous_data = cfg.task.build_anomalous_data()
    if cfg.max_size:
        reference_data = Subset(reference_data, range(cfg.max_size))
        anomalous_data = Subset(anomalous_data, range(cfg.max_size))
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
