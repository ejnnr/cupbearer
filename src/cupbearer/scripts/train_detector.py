from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task

from . import eval_detector


def main(
    task: Task,
    detector: AnomalyDetector,
    **train_kwargs,
):
    detector.set_model(task.model)

    detector.train(
        trusted_data=task.trusted_data,
        untrusted_data=task.untrusted_train_data,
        **train_kwargs,
    )
    path = detector.save_path
    if path:
        detector.save_weights(path / "detector")
        eval_detector(detector=detector, task=task, pbar=True)
