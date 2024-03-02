from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task
from cupbearer.utils.scripts import script


@script
def main(
    task: Task,
    detector: AnomalyDetector,
    pbar: bool = False,
):
    detector.set_model(task.model)

    detector.eval(
        train_dataset=task.trusted_data,
        test_dataset=task.test_data,
        pbar=pbar,
    )
