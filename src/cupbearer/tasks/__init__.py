from ._config import TaskConfig as TaskConfig
from ._config import TaskConfigBase as TaskConfigBase
from .adversarial_examples import AdversarialExampleTask
from .backdoor_detection import BackdoorDetection
from .toy_features import ToyFeaturesTask

TASKS = {
    "backdoor": BackdoorDetection,
    "adversarial_examples": AdversarialExampleTask,
    "toy_features": ToyFeaturesTask,
}
