from pathlib import Path

from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task


def main(
    task: Task,
    detector: AnomalyDetector,
    save_path: Path | str | None,
    pbar: bool = False,
    batch_size: int = 1024,
    layerwise: bool = False,
):
    detector.set_model(task.model)

    return detector.eval(
        dataset=task.test_data,
        pbar=pbar,
        save_path=save_path,
        batch_size=batch_size,
        layerwise=layerwise,
    )
