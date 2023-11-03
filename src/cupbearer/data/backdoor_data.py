# This needs to be in a separate file from backdoors.py because of circularity issues
# with the config groups. See __init__.py.
import warnings
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.data._shared import Resize, ToNumpy, Transform
from cupbearer.data.backdoors import Backdoor, WanetBackdoor
from cupbearer.utils.config_groups import config_group


@dataclass
class BackdoorData(DatasetConfig):
    original: DatasetConfig = config_group(DatasetConfig)
    backdoor: Backdoor = config_group(Backdoor)
    augment_backdoor: Optional[bool] = None  # whether or not to apply
    # backdoor before data
    # augmentation

    def __post_init__(self):
        if self.apply_before_augmentation is None:
            if isinstance(self.backdoor, WanetBackdoor):
                self.augment_backdoor = True
            else:
                self.augment_backdoor = False

    @property
    def num_classes(self):
        return self.original.num_classes

    def get_transforms(self) -> list[Transform]:
        # We can't set this in __post_init__, since then the backdoor would be part of
        # transforms in the config that's stored to disk. If we then load this config,
        # another backdoor would be added to the transforms.
        transforms = []
        transforms += self.original.get_transforms()
        transforms += super().get_transforms()

        # Insert backdoor
        if self.augment_backdoor:
            # Insert first or after ToNumpy if that is used
            for i_transform, transform in enumerate(transforms):
                if isinstance(transform, ToNumpy):
                    backdoor_index = i_transform + 1
                    break
                elif not isinstance(transform, Resize):
                    warnings.warn(
                        f"{type(transform)} is not a known non-augmentation transform,"
                        "backdoor might not be applied in the right place"
                    )

        else:
            # Insert last
            backdoor_index = len(transforms)
        transforms.insert(backdoor_index, self.backdoor)
        return transforms

    def _build(self):
        return self.original._build()
