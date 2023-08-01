from cupbearer.utils.config_groups import register_config_group

from ._config import TaskConfig as TaskConfig
from ._config import TaskConfigBase
from .adversarial_examples import AdversarialExampleTask
from .backdoor_detection import BackdoorDetection

TASKS = {
    "backdoor": BackdoorDetection,
    "adversarial_examples": AdversarialExampleTask,
}

register_config_group(TaskConfigBase, TASKS)
