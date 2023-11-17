from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import run


def main(cfg: Config):
    # Init
    train_data = cfg.task.build_train_data()
    test_data = cfg.task.build_test_data()
    # train_data[0] is the first sample, which is (input, ...), so we need another [0]
    example_input = train_data[0][0]
    model = cfg.task.build_model(input_shape=example_input.shape)
    detector = cfg.detector.build(
        model=model,
        save_dir=cfg.dir.path,
    )

    # Evaluate detector
    detector.eval(
        train_dataset=train_data,
        test_dataset=test_data,
        pbar=cfg.pbar,
    )


if __name__ == "__main__":
    run(main, Config)
