# High-level structure of `cupbearer`
In this document, we'll go over all the subpackages of `cupbearer` to see what role
they play and how to extend them. For more details of extending `cupbearer`, see
the other documentation files on specific subpackages.

## Helper subpackages
### `cupbearer.data`
The `data` package contains implementations of basic datasets, transforms,
and specialized datasets (e.g. datasets consisting only of adversarial examples).

Using this subpackage is optional, you can define tasks directly using standard
pytorch `Dataset`s.

### `cupbearer.models`
Unlike the `data` package, you have to use the `models` package at the moment.
The main reason for this is that many mechanistic anomaly detectors need access
to the model's activations. Using the implementations from the `models` package
ensures a consistent way to get activations from models. As long as you don't want
to add new model architectures, most of the details of this package won't matter.

In the future, we'll likely deprecate the `HookedModel` interface and just support
standard `torch.nn.Module`s via pytorch hooks.

### `cupbearer.utils`
The `utils` package contains some miscallaneous helper functions. Most of these are
mainly for internal usage, but see the example notebooks for helpful ones.

## Tasks
The `tasks` package contains the `Task` class, which is the interface any
task needs to implement, as well as all the existing tasks. To add a new task,
you can either inherit `Task` or simply write a function that returns a `Task` instance.

Often, you'll also need to implement a new type of dataset or model for your task.
That code probably belongs in the `data` and `model` packages,
though sometimes it's a judgement call.

See [adding_a_task.md](adding_a_task.md) for more details.

## Detectors
The `detectors` package is similar to `tasks`, but for anomaly detectors. The key
interface is `AnomalyDetector`.

See [adding_a_detector.md](adding_a_detector.md) for more details.

## Scripts
The `scripts` package contains Python functions for running common workflows.
Two scripts are meant to be used by all detectors/tasks:
- `train_detector` trains a detector on a task and saves the trained detector to disk.
- `eval_detector` evaluates a stored (or otherwise specified) detector and evaluates
  it on a task.

All other scripts are helper scripts for specific tasks or detectors. For example,
most tasks will need a script to train the model to be analyzed, and perhaps to prepare
the dataset.
