"""
An abstract class for motion simulators
"""
from typing import Union, List
from abc import ABC, abstractmethod

import numpy as np


class Simulator(ABC):
    """
    motion_noise and measurement_noise are the standard deviations of the Gaussian noise
    """
    def __init__(self, motion_noise: Union[float, np.ndarray] = 0.1,
                 measurement_noise: Union[float, np.ndarray] = 0.4):
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

    @abstractmethod
    def motion(self) -> None:
        """
        Simulate motion
        """
        pass

    @abstractmethod
    def measurement(self):
        """
        Simulate measurement
        """
        pass
