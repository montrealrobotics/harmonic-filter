"""
Base abstarct class for FFT transforms
"""
import numpy as np
from abc import ABC, abstractmethod


class FFTBase(ABC):
    def __init__(self, bandwidth: int, oversampling_factor: int):
        self.bandwidth: int = bandwidth
        self.oversampling_factor: int = oversampling_factor
        self.oversampled_b: int = self.bandwidth * self.oversampling_factor

    @abstractmethod
    def analyze(self, f: np.ndarray, oversample: bool) -> np.ndarray:
        """
        Analyze a signal using the fourier transform
        :param f: signal to analyze
        :param oversample: whether to oversample the signal
        :return: fourier coefficients
        """
        pass

    @abstractmethod
    def synthesize(self, f_hat: np.ndarray, oversample: bool) -> np.ndarray:
        """
        Synthesize a signal using the fourier transform
        :param f_hat: fourier coefficients to synthesize
        :param oversample: whether to oversample the signal
        :return: synthesized signal
        """
        pass
