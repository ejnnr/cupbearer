# Creating new scripts
You don't need to implement any scripts your new task or detector needs using the
interface described in this document. However, it's designed to work well with the
rest of `cupbearer` and probably makes sense for most cases.

As an overview, here's how to create a new script:
1. Put a python file in `scripts` with some function `my_function`.
2. The only argument to `my_function` should be an object of a dataclass `MyConfig`
   that inherits from `utils.scripts.ScriptConfig`.
3. The definition of `MyConfig` needs to be placed in its own file.
4. Use `utils.scripts.run(my_function, MyConfig)` in the python file to run the script.
5. Now you'll be able to run the script from the command line using `python -m cupbearer.scripts.my_file`.

The rest of this doc goes into some background to understand how scripts work
in `cupbearer`.

## Example walkthrough
Let's look at `eval_detector.py`:
```python
from cupbearer.scripts.conf.eval_detector_conf import Config
from cupbearer.utils.scripts import run
from torch.utils.data import Subset


def main(cfg: Config):
    reference_data = cfg.task.build_reference_data()
    anomalous_data = cfg.task.build_anomalous_data()
    if cfg.max_size:
        reference_data = Subset(reference_data, range(cfg.max_size))
        anomalous_data = Subset(anomalous_data, range(cfg.max_size))
    model = cfg.task.build_model()
    params = cfg.task.build_params()
    detector = cfg.detector.build(model=model, params=params, save_dir=cfg.dir.path)

    detector.eval(
        normal_dataset=reference_data,
        anomalous_datasets={"anomalous": anomalous_data},
    )


if __name__ == "__main__":
    run(main, Config, save_config=False)
```
There are two key things to note here:
- We have a function `main` that takes a single argument of type `Config`. (The name of `main` doesn't matter.)
- If the script is run as the main file, we call `run(main, Config)`.

Actually, in this case, we also have `save_config=False` in the call to `run`. By default,
`run` will save the config as a yaml file, which this flag disables.

Here is the definition of `Config`, in `conf/eval_detector_conf.py`:
```python
@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    max_size: Optional[int] = None

    def _set_debug(self):
        super()._set_debug()
        self.max_size = 2
```
A few things to note:
- `Config` inherits from `ScriptConfig`. All script configurations should do this.
- `Config` is a dataclass, as all configs should be.
- For the fields that are themselves dataclasses, we use `config_group` as a default.
  This lets users set these fields from the command line (where you otherwise couldn't
  pass dataclasses as values). A config group is basically a dictionary mapping from
  names (that users use on the CLI) to subclasses of some base class. For example,
  `config_group(DetectorConfig)` means that users can choose any of the registered
  detectors. If this detector has config options, these can also be set.
  Config groups are discussed in more detail in [configuration.md](configuration.md).
- There's a `_set_debug` method. This is a special method that's called when the
  `--debug` flag is passed to the script. It should set all values where this makes
  sense to values that lead to a fast run. (For example, this flag is always used
  in unit tests.) The `super()._set_debug()` call is important, since it ensures
  that `_set_debug` is called recusively on all fields that support it.
  Again see [configuration.md](configuration.md) for more details.

## The `Config` definition needs to be in its own file
There is currently a technical limitation: the definition of the `Config` class
mustn't be in the same file as the script that users will call from the CLI. That's
why all the configs are in the `conf` folder.

The reason for this is that serializing a configuration dataclass to yaml stores
the full path of the dataclass (in order to reliably deserialize it later). If the
dataclass is defined in the main script, that path will be `__main__.Config`, which
can then not be restored from a different script.

## `ScriptConfig`
As mentioned above, all configs for scripts should inherit from `ScriptConfig`.
Let's look at `ScriptConfig` to understand the effects of that:
```python
@dataclass(kw_only=True)
class ScriptConfig(BaseConfig):
    seed: int = 0
    dir: DirConfig = mutable_field(DirConfig)
    debug: bool = field(action="store_true")
    debug_with_logging: bool = field(action="store_true")

    ...
```
(See [configuration.md](configuration.md) for more details on `BaseConfig`.)

This is where the `debug` flag mentioned above is defined (the `field` here
is from `simple_parsing` and extends `dataclasses.field`). Apart from that, there's
a `seed` field, since basically every script will need that.

Perhaps most interesting is the `dir` field. This is a `DirConfig`, which has three
fields:
```python
@dataclass(kw_only=True)
class DirConfig(BaseConfig):
    base: Optional[str] = None
    run: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    full: Optional[str] = None
```
By default, `base` and `full` are `None`. This means that nothing will be logged to disk.
If `full` is set, then the path will always be `full`, no matter what `base` and `run`
are. Otherwise, if `base` is set, the path will be `base/run`. This can be useful if you
want to automatically generate new directories for each run without naming them all.
For example,
```bash
python -m cupbearer.scripts.train_detector --dir.base logs/train_detector ...
```
would create a new directory `logs/train_detector/<current date & time>/` for each run.

While typically, `ScriptConfig.dir` is meant to be a newly created logging directory,
it can also sometimes take on the role of an input directory. For example,
```bash
python -m cupbearer.scripts.eval_detector --dir.full logs/train_detector/... --detector from_run ...
```
would load a detector from the directory `logs/train_detector/...` and evaluate it
(since the `from_run` option for the detector config group uses the `dir` argument).

How directories are handled is one of the places that seems most likely to change,
so try not to rely too much on the current version.