# ruff: noqa: F401
from .conf.eval_classifier_conf import Config as EvalClassifierConfig
from .conf.eval_detector_conf import Config as EvalDetectorConfig
from .conf.train_classifier_conf import Config as TrainClassifierConfig
from .conf.train_detector_conf import Config as TrainDetectorConfig
from .eval_classifier import main as eval_classifier
from .eval_detector import main as eval_detector
from .train_classifier import main as train_classifier
from .train_detector import main as train_detector
