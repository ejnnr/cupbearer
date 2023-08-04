from dataclasses import MISSING
from typing import Callable, Optional, TypeVar

import simple_parsing
from cupbearer.utils.utils import BaseConfig

_CONFIG_GROUPS = {}

ConfigType = TypeVar("ConfigType", bound=BaseConfig)


def register_config_group(base: type[ConfigType], options: dict[str, type[ConfigType]]):
    if base in _CONFIG_GROUPS:
        raise ValueError(
            f"Config group {base} already registered "
            f"(options: {', '.join(_CONFIG_GROUPS[base].keys())})."
        )
    _CONFIG_GROUPS[base] = options


def register_config_option(base: type[ConfigType], name: str, option: type[ConfigType]):
    if base not in _CONFIG_GROUPS:
        _CONFIG_GROUPS[base] = {}
    _CONFIG_GROUPS[base][name] = option


def config_group(base: type, default_factory: Optional[Callable] = None):
    return simple_parsing.subgroups(
        _CONFIG_GROUPS[base], default_factory=default_factory or MISSING  # type: ignore
    )
