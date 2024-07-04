"""
Code adapted from https://github.com/AMLab-Amsterdam/lie_learn
"""
import numpy as np
from numpy.fft import rfft, irfft, fftshift, ifftshift
from src.spectral.base_fft import FFTBase


class S1FFT(FFTBase):
    """
    The Fast Fourier Transform on the Circle.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyze(self, f: np.ndarray, oversample: bool = False) -> np.ndarray:
        """
        Compute the Fourier Transform of the discretely sampled function f : T^1 -> C.

        Let f : T^1 -> C be a band-limited function on the circle.
        The samples f(theta_k) correspond to points on a regular grid on the circle, as returned by spaces.T1.linspace:
        theta_k = 2 pi k / N
        for k = 0, ..., N - 1

        This function computes
        \hat{f}_n = (1/N) \sum_{k=0}^{N-1} f(theta_k) e^{-i n theta_k}
        which, if f has band-limit less than N, is equal to:
        \hat{f}_n = \int_0^{2pi} f(theta) e^{-i n theta} dtheta / 2pi,
                  = <f(theta), e^{i n theta}>
        where dtheta / 2pi is the normalized Haar measure on T^1, and < , > denotes the inner product on Hilbert space,
        with respect to which this transform is unitary.

        The range of frequencies n is -floor(N/2) <= n <= ceil(N/2) - 1

        :param f: signal to analyze
        :param oversample: whether to oversample the signal
        :return: S1 fourier transform of signal
        """
        # The numpy FFT returns coefficients in a different order than we want them,
        # and using a different normalization.
        b = self.oversampled_b if oversample else self.bandwidth
        f_hat = rfft(f, axis=0)
        f_hat = fftshift(f_hat, axes=0)
        return f_hat / b

    def synthesize(self, f_hat: np.ndarray, oversample: bool = False) -> np.ndarray:
        """
        Compute the inverse / synthesis Fourier transform of the function f_hat : Z -> C.
        The function f_hat(n) is sampled at points in a limited range -floor(N/2) <= n <= ceil(N/2) - 1

        This function returns
        f[k] = f(theta_k) = sum_{n=-floor(N/2)}^{ceil(N/2)-1} f_hat(n) exp(i n theta_k)
        where theta_k = 2 pi k / N
        for k = 0, ..., N - 1

        :param f_hat: fourier transform to synthesize
        :param oversample: whether to oversample the signal
        :return: synthesized signal in S1 group
        """
        b = self.oversampled_b if oversample else self.bandwidth
        f_hat = ifftshift(f_hat * b, axes=0)
        f = irfft(f_hat, axis=0, n=b)
        return f

    def analyze_naive(self, f: np.ndarray, oversample: bool = False) -> np.ndarray:
        b = self.oversampled_b if oversample else self.bandwidth
        f_hat = np.zeros_like(f)
        for n in range(self.bandwidth):
            for k in range(self.bandwidth):
                theta_k = k * 2 * np.pi / self.bandwidth
                f_hat[n] += f[k] * np.exp(-1j * n * theta_k)
        return fftshift(f_hat / self.bandwidth, axes=0)
