from abc import ABC
from dataclasses import dataclass


class DataFormat(ABC):
    pass


@dataclass
class TensorDataFormat(DataFormat):
    shape: tuple[int, ...] | list[int]


class TextDataFormat(DataFormat):
    pass
