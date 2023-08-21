from cupbearer.utils.config_groups import register_config_group

from ._config import TaskConfig as TaskConfig
from ._config import TaskConfigBase
from .adversarial_examples import AdversarialExampleTask
from .backdoor_detection import BackdoorDetection
from .toy_features import ToyFeaturesTask

TASKS = {
    "backdoor": BackdoorDetection,
    "adversarial_examples": AdversarialExampleTask,
    "toy_features": ToyFeaturesTask,
}

register_config_group(TaskConfigBase, TASKS)
