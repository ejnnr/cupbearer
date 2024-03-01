import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

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
    # if cfg.path:
    #     cfg.path.mkdir(parents=True, exist_ok=True)
    #     if save_config:
    #         # TODO: replace this with cfg.save if/when that exposes save_dc_types.
    #         # Note that we need save_dc_types here even though `BaseConfig` already
    #         # enables that, since `save` calls `to_dict` directly, not `obj.to_dict`.
    #         simple_parsing.helpers.serialization.serializable.save(
    #             cfg,
    #             cfg.path / "config.yaml",
    #             save_dc_types=True,
    #             sort_keys=False,
    #         )
    pass


T = TypeVar("T")
