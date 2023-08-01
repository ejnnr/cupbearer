# High-level structure of `cupbearer`
In this document, we'll go over all the subpackages of `cupbearer` to see what role
they play and how to extend them. For more details of extending `cupbearer`, see
the other documentation files on specific subpackages.

## Configuration
Different parts of `cupbearer` interface with each other through many configuration
dataclasses. This makes it easy to use newly added parts from the existing command
line scripts, without modifying them. For example, the main interface for an
anomaly detector is the `DetectorConfig` ABC. If a new detector defines a subclass
of this ABC and registers it, it will automatically be available in the `train_detector`
and `eval_detector` scripts.

Many of the configuration dataclass ABCs have one or several `build()` methods that
create the actual object of interest based on the configuration. For example,
the `DetectorConfig` ABC has an abstract `build()` method that must return an
`AnomalyDetector` instance.

See [configuration.md](configuration.md) for more details on the configuration
dataclasses and what to keep in mind when writing your own.

## Helper subpackages
### `cupbearer.data`
The `data` package contains implementations of basic datasets, transforms,
and specialized datasets (e.g. datasets consisting only of adversarial examples).
The key interface is the `DatasetConfig` class. It has a `build()` method that
needs to return a pytorch `Dataset` instance.

In principle, you don't need to use the `DatasetConfig` interface (or anything
from the `data` package) to implement new tasks or detectors. Tasks and detectors
just pass `Dataset` instances between each other. But unless you have a good reason
to avoid the `DatasetConfig` interface, it's best to use it since it already works
with the scripts and you get some features such as configuring transforms for free.

### `cupbearer.models`
Unlike the `data` package, you have to use the `models` package at the moment.
The main reason for this is that many mechanistic anomaly detectors need access
to the model's activations. Using the implementations from the `models` package
ensures a consistent way to get activations from models. As long as you don't want
to add new model architectures, most of the details of this package won't matter.

For now, only linear computational graphs are supported, i.e. each model needs to
be a fixed sequence of computational steps performed one after the other
(like a `Sequential` module in many deep learning frameworks). A `Computation`
is just a type alias for such as sequence of steps. The `Model` class takes such a
`Computation` and is itself a `flax.linen.Module` that implements the computation.
The main thing it does on top of `flax.linen.Sequential` is that it can also return
all the activations of the model. It also has a function for plotting the architecture
of the model.

Similar to the `DataConfig` interface, there's a `ModelConfig` with a `build()`
method that returns a `Model` instance.

### `cupbearer.utils`
The `utils` package contains many miscallaneous helper functions. You probably won't
interact with these too much, but here are a few that it may be good to know about:
- `utils.trainer` contains a `Trainer` class that's a very simple version of pytorch
  lightning for flax. You certainly don't need to use this in any scripts you add,
  but it may save you some boilerplate. NOTE: we might deprecate this in the future
  and replace it with something like `elegy`.
- `utils.utils.save` and `utils.utils.load` can save and store pytrees. They use the
  `orbax` checkpointer under the hood, but add some hacky support for saving/loading
  types.

We'll cover a few more functions from the `utils` package when we talk about scripts.

## Tasks
The `tasks` package contains the `TaskConfigBase` ABC, which is the interface any
task needs to implement, as well as all the existing tasks. To add a new task:
1. Create a new module or subpackage in `tasks`, where you implement a new class
   that inherits `TaskConfigBase`.
2. Add your new class to the `TASKS` dictionary in `tasks/__init__.py`.

Often, you'll also need to implement a new type of dataset or model.
That code probably belongs in the `data` and `model` packages,
though sometimes it's a judgement call.

See [adding_a_task.md](adding_a_task.md) for more details.

## Detectors
The `detectors` package is similar to `tasks`, but for anomaly detectors. In addition
to the `DetectorConfig` interface, it also contains an `AnomalyDetector` ABC, which
any detection method needs to subclass for its actual implementation.

See [adding_a_detector.md](adding_a_detector.md) for more details.

## Scripts
The `scripts` package contains command line scripts and their configurations.
Two scripts are meant to be used by all detectors/tasks:
- `train_detector` trains a detector on a task and saves the trained detector to disk.
- `eval_detector` evaluates a stored (or otherwise specified) detector and evaluates
  it on a task.

All other scripts are helper scripts for specific tasks or detectors. For example,
most tasks will need a script to train the model to be analyzed, and perhaps to prepare
the dataset.

There's a lot more to be said about scripts, see the [README](../README.md) for a brief
overview of *running* scripts, and [adding_a_script.md](adding_a_script.md) for details
on writing new scripts.