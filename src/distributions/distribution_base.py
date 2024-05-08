"""
An abstract class for probability distributions
"""
from typing import Type, Tuple

import numpy as np
from abc import ABC, abstractmethod

from src.spectral.base_fft import FFTBase


class HarmonicExponentialDistribution(ABC):
    def __init__(self, samples: np.ndarray, fft: FFTBase):
        # Etas
        self._eta: np.ndarray = None
        self._M: np.ndarray = None
        # Store energy and prob samples
        self._energy: np.ndarray = None
        self._prob: np.ndarray = None
        # Grid samples
        self.samples: np.ndarray = samples
        # Moments of the distribution and normalizing constant
        self.moments: np.ndarray = None
        self.l_n_z: float = None
        # FFT object
        self.fft: Type[FFTBase] = fft

    def from_samples(self) -> None:
        """
        Set up the distribution from samples
        :return : none
        """
        # Compute energy of samples
        energy = self.compute_energy(self.samples)
        self.eta = self.fft.analyze(energy)
        self.energy = self.fft.synthesize(self.eta)

    @classmethod
    def from_eta(cls, eta: np.ndarray, fft: FFTBase) -> Type['HarmonicExponentialDistribution']:
        """
        Create a distribution from log eta - it does not compute energy from samples
        :param eta: log eta parameter
        :param fft: fft object
        :return: distribution
        """
        dist = cls(samples=None, fft=fft)
        dist._eta = eta
        return dist

    @classmethod
    def from_M(cls, M: np.ndarray, fft: FFTBase) -> Type['HarmonicExponentialDistribution']:
        """
        Create a distribution from M - it does not compute energy from samples
        :param M: M parameter
        :param fft: fft object
        :return: distribution
        """
        dist = cls(samples=None, fft=fft)
        # This will update M and also unnormalize to compute M
        dist.M = M
        # We need to compute M again to get the correct M as the result from convolution is unnormalized
        dist.normalize()
        return dist

    @abstractmethod
    def normalize(self):
        """
        Updated moments and log partition constant and compute etas
        :return: none
        """
        pass

    @abstractmethod
    def compute_eta(self):
        """
        Compute eta from eta
        :return: none
        """
        pass

    @abstractmethod
    def product(self, dist: Type['HarmonicExponentialDistribution']) -> Type['HarmonicExponentialDistribution']:
        """
        Product of two distribution and update canonical parameters
        :param dist: distribution to multiply with
        :return: distribution with updated eta
        """
        pass

    @abstractmethod
    def convolve(self, dist: Type['HarmonicExponentialDistribution']) -> Type['HarmonicExponentialDistribution']:
        """
        Convolution of two distribution and update canonical parameters in log space
        :param dist: distribution to convolve with
        :return: distribution with updated eta
        """
        pass

    @abstractmethod
    def compute_moments_lnz(self, eta: np.ndarray, update: bool = True) -> Tuple[np.ndarray, float]:
        """
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: none
        """
        pass

    @abstractmethod
    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the energy of the distribution
        :param t: grid to evaluate the distribution
        :return: probability
        """
        pass

    @property
    def M(self) -> np.ndarray:
        # Compute M if it is None
        if self._M is None:
            self.normalize()
        return self._M

    @M.setter
    def M(self, M: np.ndarray) -> None:
        self._M = M.copy()
        # Unnormalize to compute eta
        if self._eta is None:
            self.compute_eta()

    @property
    def eta(self) -> np.ndarray:
        """
        When eta is set, reset eta
        """
        return self._eta

    @eta.setter
    def eta(self, eta: np.ndarray) -> None:
        """
        When eta is set, reset eta
        """
        self._eta = eta.copy()

    @property
    def energy(self) -> np.ndarray:
        if self._energy is None:
            self._energy = self.fft.synthesize(self.eta).real
        return self._energy

    @energy.setter
    def energy(self, energy: np.ndarray) -> None:
        self._energy = energy.copy()

    @property
    def prob(self) -> np.ndarray:
        if self._prob is None:
            self._prob = self.fft.synthesize(self.M).real
            # Zero negative values
            self._prob = np.where(self._prob > 0, self._prob, 1e-8)
        return self._prob

    @prob.setter
    def prob(self, prob: np.ndarray) -> None:
        self._prob = prob.copy()
