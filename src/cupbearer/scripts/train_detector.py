from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task
from cupbearer.utils.scripts import script
from cupbearer.utils.train import TrainConfig
from cupbearer.utils.utils import BaseConfig

from . import EvalDetectorConfig, eval_detector


@script
def main(
    task: Task,
    detector: AnomalyDetector,
    num_classes: int,
    train: BaseConfig | None = None,
    seed: int = 0,
):
    if train is None:
        train = TrainConfig()
    detector.set_model(task.model)

    detector.train(
        trusted_data=task.trusted_data,
        untrusted_data=task.untrusted_train_data,
        num_classes=num_classes,
        train_config=train,
    )
    path = detector.save_path
    if path:
        detector.save_weights(path / "detector")
        eval_cfg = EvalDetectorConfig(
            detector=detector,
            task=task,
            seed=seed,
        )
        eval_detector(eval_cfg)
