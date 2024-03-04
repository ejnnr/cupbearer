import pytest
from cupbearer import data, models
from cupbearer.scripts import (
    eval_classifier,
    train_classifier,
)
from cupbearer.scripts.conf import (
    eval_classifier_conf,
    train_classifier_conf,
)


@pytest.fixture(scope="module")
def measurement_predictor_path(module_tmp_path):
    cfg = train_classifier_conf.DebugConfig(
        model=models.TamperTransformerConfig(name="pythia-14m"),
        train_data=data.TamperingDataConfig(name="redwoodresearch/diamonds-seed0"),
        task="multilabel",
        path=module_tmp_path,
    )
    train_classifier(cfg)

    assert (module_tmp_path / "config.yaml").is_file()
    assert (module_tmp_path / "checkpoints" / "last.ckpt").is_file()
    assert (module_tmp_path / "tensorboard").is_dir()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(measurement_predictor_path):
    cfg = eval_classifier_conf.DebugConfig(
        path=measurement_predictor_path,
        data=data.TamperingDataConfig(
            name="redwoodresearch/diamonds-seed0", train=False
        ),
    )

    eval_classifier(cfg)

    assert (measurement_predictor_path / "eval.json").is_file()
