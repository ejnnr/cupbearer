# Adding a new task

The only component that a task absolutely needs is an implementation of the
`TaskConfigBase` abstract class:
```python
class TaskConfigBase(BaseConfig, ABC):
    @abstractmethod
    def build_reference_data(self) -> Dataset:
        pass

    @abstractmethod
    def build_model(self) -> Model:
        pass

    def build_params(self):
        return None

    @abstractmethod
    def build_anomalous_data(self) -> Dataset:
        pass
```
If your config has any parameters, you should use a dataclass to set them. E.g.
```python
@dataclass
class MyTaskConfig(TaskConfigBase):
    my_required_param: str
    my_optional_param: int = 42

    ...
```
This will automagically let you override these parameters from the command line
(and any parameters without default values will be required).

`build_reference_data` and `build_anomalous_data` both need to return `pytorch` `Dataset`s.
`build_model` needs to return a `models.Model`, which is a special type of `flax.linen.Module`.
`build_params` can return a parameter dict for the returned `Model` (if `None`, the model
will be randomly initialized, which is usually not what you want).

In practice, the datasets and the model will have to come from somewhere, so you'll
often implement a few things in addition to the task config class. There are predefined
interfaces for datasets and models, and if possible I suggest using those (either
using their existing implementations, or adding your own). For example, consider
the adversarial example task:
```python
@dataclass
class AdversarialExampleTask(TaskConfigBase):
    run_path: Path

    def __post_init__(self):
        self._reference_data = TrainDataFromRun(path=self.run_path)
        self._anomalous_data = AdversarialExampleConfig(run_path=self.run_path)
        self._model = StoredModel(path=self.run_path)

    def build_anomalous_data(self) -> Dataset:
        return self._anomalous_data.build_dataset()

    def build_model(self) -> Model:
        return self._model.build_model()

    def build_params(self) -> Model:
        return self._model.build_params()

    def build_reference_data(self) -> Dataset:
        return self._reference_data.build_dataset()
```
This task only has one parameter, the path to the training run of a base model.
It then uses the training data of that run as reference data, and an adversarial
version of it as anomalous data. The model is just the trained base model, loaded
from disk.

You can also add new scripts in the `scripts` directory, to generate the datasets
and/or train the model. For example, the adversarial examples task has an
associated script `make_adversarial_examples.py`. (To get the model, we can simply
use the existing `train_classifier.py` script.)

There's no formal connection between scripts and the rest of the library---you can
leave it up to users to run the necessary preparatory scripts before using your new
task. But if feasible, you may want to automate this. For example, the `AdversarialExampleDataset`
automatically runs `make_adversarial_examples.py` if the necessary files are not found.

Finally, you need to register your task to make it accessible from the command line
in the existing scripts. Simply add the task config class to the `TASKS` dict in `tasks/__init__.py`
(with an arbitrary name as the key).

Then you should be able to run commands like
```bash
python -m cupbearer.scripts.train_detector --task my_task --detector my_detector --task.my_required_param foo
```
