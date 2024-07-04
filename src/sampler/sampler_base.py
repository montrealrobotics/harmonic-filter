"""
An abstract class to sample groups
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, n_samples: int):
        self.n_samples: int = n_samples
        self.samples: np.ndarray = None

    @abstractmethod
    def sample(self):
        """
        Grid samples from the group
        :return: samples
        """
        pass
