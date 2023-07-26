import functools
from hydra.core.hydra_config import HydraConfig
import re
from hydra import TaskFunction
import hydra
from hydra.core.config_store import ConfigStore
from hydra.experimental.callback import Callback
import shutil
from typing import Any, Optional, Type
from hydra.utils import get_original_cwd


import os
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def original_relative_path(path: str | Path) -> Path:
    """Converts a path to be relative to the original (pre-hydra) working directory."""
    new_cwd = os.getcwd()
    abs_path = Path(new_cwd) / Path(path)
    original_cwd = get_original_cwd()
    rel_path = os.path.relpath(abs_path, original_cwd)
    return Path(rel_path)


class CheckOutputDirExistsCallback(Callback):
    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        # TODO: this doesn't actually work
        hydra_cfg = HydraConfig.get()
        if not hydra_cfg.output_subdir:
            # For some jobs, like eval_detector, there's no dedicated output dir.
            # We want to skip checks for these.
            return
        # This hook isn't really necessary given that on_job_start below takes
        # care of similar things. But on non-multiruns, this hook will be called
        # *before* the .hydra dir is overwritten, so it gives us more safety at
        # least for those cases.
        if os.path.exists(hydra_cfg.run.dir):
            if config.get("overwrite_output", False):
                logger.info("Overwriting output dir")
                shutil.rmtree(hydra_cfg.run.dir)
            else:
                raise BaseException(
                    "Output dir already exists! Use +overwrite_output=true to overwrite"
                )

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ):
        """Check that the output dir is empty, except for the .hydra dir and logfile.

        This hook is needed for multirun jobs: the previous one won't trigger,
        but we don't want to use on_multirun_start because that would only allow
        checking whether the entire sweep directory exists.
        Instead, we'd like overwrite checks on a job-basis.

        TODO: The downside of this approach is that this hook is only called after the
        .hydra dir has already been created and potentially overwrote the existing one.
        So there's a danger of data loss even with overwrite_output=false.
        """
        hydra_cfg = HydraConfig.get()
        if not hydra_cfg.output_subdir:
            # For some jobs, like eval_detector, there's no dedicated output dir.
            # We want to skip checks for these.
            return
        path = Path(hydra_cfg.runtime.output_dir)
        files = os.listdir(path)
        if not set(files) <= {".hydra", f"{hydra_cfg.job.name}.log"}:
            if config.get("overwrite_output", False):
                logger.info("Overwriting output dir")
                # Don't remove the .hydra dir, that's from the new run and already
                # overwrites the old one anyway.
                # Do delete the log file because otherwise hydra will append to it.
                # (There shouldn't be anything in the logfile yet.)
                files.remove(".hydra")
                for file in files:
                    os.remove(path / file)
            else:
                raise BaseException(
                    "Output dir already exists! Use +overwrite_output=true to overwrite"
                    f" (.hydra dir at {path / '.hydra'} has already been overwritten!)"
                )


# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


def get_grid_subdir(override_dirname: str) -> str:
    # override_dirname is a string of the form
    # +experiment=mnist_cnn|+attack=pixel_backdoor|batch_size=32
    # We want to format this as mnist_cnn,pixel_backdoor,batch_size=32

    # Split the string into parts by comma
    parts = override_dirname.split("|")

    # Remove the '+' character and get the value after '=' for each part
    formatted_parts = [part.replace("+", "").split("=") for part in parts]
    config = {
        name: value
        for name, value in formatted_parts
        # Things in the hydra.launcher namespace are just Slurm arguments,
        # no need to include them in directory names
        if not name.startswith("hydra.launcher")
    }

    # First, create the experiment part of the subdir string
    subdir = config["experiment"]
    del config["experiment"]

    # Next, add the attack part of the subdir string
    subdir += "," + config["attack"]
    del config["attack"]

    # Avoid adding an extra comma if the config is empty
    if config:
        # Sort config by keys to get consistent directory names:
        config = dict(sorted(config.items()))
        subdir += "," + ",".join(f"{name}={value}" for name, value in config.items())

    return subdir


def register_resolvers():
    # This function may be called multiple times because it's called in setup_hydra,
    # which is called outside of main blocks, and so executed when importing script
    # modules. In that case, OmegaConf raises an error, so we just skip registration.
    try:
        OmegaConf.register_new_resolver("escape", lambda x: x.replace("/", "_"))
        OmegaConf.register_new_resolver("get_grid_subdir", get_grid_subdir)
    except ValueError as e:
        # Make sure we only skip the error we expect
        if "is already registered" not in str(e):
            raise


def setup_hydra(config_name):
    # Hack to get more than one config directory.
    # See https://github.com/facebookresearch/hydra/issues/2001#issuecomment-1032968874
    cs = ConfigStore.instance()

    SHARED_PATH = Path(__file__).parent / "conf" / "_shared"

    custom_path_list = [str(SHARED_PATH)]

    config = {
        "defaults": ["main", "_self_"],
        "hydra": {"searchpath": custom_path_list},
    }
    cs.store(name=config_name, node=config)

    register_resolvers()


def get_subconfig(path: str, name: str, expected_type: Optional[Type] = None):
    run = Path(path)
    cfg = OmegaConf.load(
        hydra.utils.to_absolute_path(str(run / ".hydra" / "config.yaml"))
    )
    cfg = OmegaConf.to_object(cfg)

    if not hasattr(cfg, name):
        raise ValueError(f"Expected train_data to be in config, got {cfg}")

    sub_cfg = getattr(cfg, name)

    if expected_type and not isinstance(sub_cfg, expected_type):
        raise ValueError(f"Expected {name} to be a {expected_type}, got {sub_cfg}")

    return sub_cfg


# https://stackoverflow.com/a/14412901
def doublewrap(f):
    """
    A decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


CONFIG_GROUP_BASES: dict[Type, str] = {}


@doublewrap
def hydra_config_base(node: Type, group: str):
    """Decorator for registering a dataclass as the base class for some config group.

    Doesn't have any effects on hydra itself, but this info is used by
    the `@hydra_config` decorator.
    """
    CONFIG_GROUP_BASES[node] = group
    logger.debug(f"Registered config base {node} under group {group}")
    return node


@doublewrap
def hydra_config(node: Type, name: Optional[str] = None, group: Optional[str] = None):
    """Decorator for registering a structured config with hydra.

    We could also add the `@dataclass` decorator within here, but that confuses
    IDEs, so for now it seems better to be more explicit. Structured configs should
    thus be annotated as
    ```
    @hydra_config  # or e.g. @hydra_config("my_name", "my_group")
    @dataclass
    class MyConfig:
        ...
    ```

    Args:
        node: The dataclass to register.
        name: The name to register the dataclass under. If `None`, the name of the
            dataclass will be converted from CamelCase to snake_case.
        group: The config group to register the dataclass under, if any.
            Will typically be assigned automatically if the config is subclassing
            one of the config base classes (like `DatasetConfig`).
    """
    if name is None:
        name = re.sub(
            r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r"_\1", node.__name__
        ).lower()

    if group is None:
        for base, base_group in CONFIG_GROUP_BASES.items():
            if issubclass(node, base):
                group = base_group
                break

    cs = ConfigStore.instance()
    cs.store(group=group, name=name, node=node)
    logger.debug(f"Registered config {name} under group {group}")

    return node
