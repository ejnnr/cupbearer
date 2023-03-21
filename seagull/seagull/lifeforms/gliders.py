# -*- coding: utf-8 -*-

"""Gliders are lifeforms that oscillate but move while oscillating"""

# Import modules
import numpy as np
from enum import Enum

from .base import Lifeform


class Direction(Enum):
    SE = 0
    SW = 1
    NW = 2
    NE = 3


class Glider(Lifeform):
    def __init__(self, direction: Direction = Direction.SE, phase: int = 0):
        """Initialize the class.

        Args:
            direction: Direction the glider is moving in
            phase: Phase of the glider (0 to 3)
        """
        if phase not in range(4):
            raise ValueError("Phase must be an integer between 0 and 3")
        super().__init__()
        self.direction = direction
        self.phase = phase

    @property
    def layout(self) -> np.ndarray:
        # This is in the default (SE) direction:
        match self.phase:
            case 0:
                x = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
            case 1:
                x = [[1, 0, 1], [1, 1, 1], [0, 1, 0]]
            case 2:
                x = [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
            case 3:
                x = [[1, 0, 0], [0, 1, 1], [1, 1, 0]]
            case _:
                raise ValueError("Phase must be an integer between 0 and 3")

        x = np.array(x)
        # Rotate the glider counter-clockwise:
        x = np.rot90(x, -self.direction.value)

        return x


class LightweightSpaceship(Lifeform):
    def __init__(self):
        super(LightweightSpaceship, self).__init__()

    @property
    def layout(self) -> np.ndarray:
        return np.array(
            [
                [0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        )


class MiddleweightSpaceship(Lifeform):
    def __init__(self):
        super(MiddleweightSpaceship, self).__init__()

    @property
    def layout(self) -> np.ndarray:
        return np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 0],
            ]
        )
