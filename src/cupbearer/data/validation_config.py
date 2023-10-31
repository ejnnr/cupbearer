import dataclasses
from dataclasses import dataclass
from typing import Optional

from cupbearer.data._shared import DatasetConfig, NoData
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.utils import BaseConfig


@dataclass(kw_only=True)
class ValidationConfig(BaseConfig):
    # Currently these fields have the same defaults
    val: Optional[DatasetConfig] = config_group(DatasetConfig, NoData)
    clean: Optional[DatasetConfig] = config_group(DatasetConfig, NoData)
    custom: Optional[DatasetConfig] = config_group(DatasetConfig, NoData)
    backdoor: Optional[DatasetConfig] = config_group(DatasetConfig, NoData)

    def items(self) -> list[tuple[str, DatasetConfig]]:
        res = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, DatasetConfig) and not isinstance(value, NoData):
                res.append((field.name, value))

        return res
