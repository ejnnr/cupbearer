from abc import ABC
from typing import Any, Dict, Mapping, Optional


class Logger(ABC):
    """Base class for all metric loggers.

    Subclasses need to override either `_log_scalar` or `log_metrics`.
    """

    def _log_scalar(self, name: str, value: Any, step: int, **kwargs):
        raise NotImplementedError

    def log_metrics(self, metrics: Mapping[str, Any], step: int):
        for name, value in metrics.items():
            self._log_scalar(name, value, step)

    def close(self):
        pass


class DummyLogger(Logger):
    def _log_scalar(self, name: str, value: Any, step: int, **kwargs):
        pass


class ClearMLLogger(Logger):
    def __init__(self, project_name: str, task_name: str):
        super().__init__()
        # Import here instead of at the top so this isn't a hard dependency
        from clearml import Task

        # Don't seed anything here, that should be handled elsewhere
        Task.set_random_seed(None)
        self.task = Task.init(project_name=project_name, task_name=task_name)
        self.logger = self.task.get_logger()

    def _log_scalar(self, name: str, value: Any, step: int, **kwargs):
        # ClearML takes a name for a plot and then separately a name
        # for the series in that plot. For now, we just make an extra
        # plot for every series.
        return self.logger.report_scalar(name, name, value, step)

    def close(self):
        self.task.close()


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        task_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        import wandb

        wandb.init(project=project_name, name=task_name, config=config, **kwargs)
        self.logger = wandb

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        return self.logger.log(metrics, step)

    def close(self):
        self.logger.finish()
