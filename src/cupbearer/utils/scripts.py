import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import simple_parsing
from loguru import logger

from cupbearer.utils.utils import BaseConfig


@dataclass(kw_only=True)
class ScriptConfig(BaseConfig):
    seed: int = 0
    path: Optional[Path] = None
    save_config: bool = True


ConfigType = TypeVar("ConfigType", bound=ScriptConfig)


def script(
    script_fn: Callable[[ConfigType], Any],
) -> Callable[[ConfigType], Any]:
    @functools.wraps(script_fn)
    def run_script(cfg: ConfigType):
        save_cfg(cfg, save_config=cfg.save_config)
        return script_fn(cfg)

    return run_script


def save_cfg(cfg: ScriptConfig, save_config: bool = True):
    if cfg.path:
        cfg.path.mkdir(parents=True, exist_ok=True)
        if save_config:
            # TODO: replace this with cfg.save if/when that exposes save_dc_types.
            # Note that we need save_dc_types here even though `BaseConfig` already
            # enables that, since `save` calls `to_dict` directly, not `obj.to_dict`.
            simple_parsing.helpers.serialization.serializable.save(
                cfg,
                cfg.path / "config.yaml",
                save_dc_types=True,
                sort_keys=False,
            )


T = TypeVar("T")


def load_config(
    path: str | Path,
    name: Optional[str] = None,
    expected_type: Type[T] = ScriptConfig,
) -> T:
    logger.debug(f"Loading config '{name}' from {path}")
    path = Path(path)
    cfg = ScriptConfig.load(path / "config.yaml", drop_extra_fields=False)

    if name is None:
        if not isinstance(cfg, expected_type):
            raise ValueError(f"Expected config to be a {expected_type}, got {cfg}")

        return cfg

    if not hasattr(cfg, name):
        raise ValueError(f"Expected {name} to be in config, got {cfg}")

    sub_cfg = getattr(cfg, name)

    if not isinstance(sub_cfg, expected_type):
        raise ValueError(f"Expected {name} to be a {expected_type}, got {sub_cfg}")

    return sub_cfg
