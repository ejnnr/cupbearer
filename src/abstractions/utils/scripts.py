from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import simple_parsing
from abstractions.utils.utils import BaseConfig
from simple_parsing.helpers import mutable_field  # type: ignore


@dataclass(kw_only=True)
class DirConfig(BaseConfig):
    base: Optional[str] = None
    run: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    full: Optional[str] = None

    @property
    def path(self) -> Optional[Path]:
        if self.full is not None:
            return Path(self.full)
        if self.base is None:
            return None
        return Path(self.base) / self.run


@dataclass(kw_only=True)
class ScriptConfig(BaseConfig):
    seed: int = 0
    dir: DirConfig = mutable_field(DirConfig)


ConfigType = TypeVar("ConfigType", bound=ScriptConfig)


def run(
    script: Callable[[ConfigType], Any],
    cfg_type: type[ConfigType],
    save_config: bool = True,
):
    cfg = simple_parsing.parse(
        cfg_type,
        argument_generation_mode=simple_parsing.ArgumentGenerationMode.NESTED,
    )

    save_cfg(cfg, save_config=save_config)

    return script(cfg)


def save_cfg(cfg: ScriptConfig, save_config: bool = True):
    if cfg.dir.path:
        cfg.dir.path.mkdir(parents=True, exist_ok=True)
        if save_config:
            # TODO: replace this with cfg.save if/when that exposes save_dc_types.
            # Note that we need save_dc_types here even though `BaseConfig` already
            # enables that, since `save` calls `to_dict` directly, not `obj.to_dict`.
            simple_parsing.helpers.serialization.serializable.save(
                cfg, cfg.dir.path / "config.yaml", save_dc_types=True
            )


T = TypeVar("T")


def load_config(
    path: str | Path,
    name: Optional[str] = None,
    expected_type: Type[T] = ScriptConfig,
) -> T:
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
