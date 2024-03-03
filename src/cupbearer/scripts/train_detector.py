from pathlib import Path

from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task

from . import eval_detector


def main(
    task: Task,
    detector: AnomalyDetector,
    save_path: Path | str | None,
    eval_batch_size: int = 1024,
    **train_kwargs,
):
    detector.set_model(task.model)

    detector.train(
        trusted_data=task.trusted_data,
        untrusted_data=task.untrusted_train_data,
        save_path=save_path,
        **train_kwargs,
    )
    if save_path:
        save_path = Path(save_path)
        detector.save_weights(save_path / "detector")
        eval_detector(
            detector=detector,
            task=task,
            pbar=True,
            batch_size=eval_batch_size,
            save_path=save_path,
        )
