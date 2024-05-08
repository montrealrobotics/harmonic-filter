"""
Distributions defined over the circle S1
"""
from typing import Tuple, Type, List
from abc import ABCMeta

import numpy as np
from scipy.special import i0
from numpy.fft import ifftshift

from src.distributions.distribution_base import HarmonicExponentialDistribution


class S1(HarmonicExponentialDistribution, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def product(cls, dist1: Type['S1'], dist2: Type['S1']) -> Type['S1']:
        """
        Product of two distribution and update canonical parameters for S1 group
        :param dist1: first distribution
        :param dist2: second distribution to multiply with
        :return: S1 distribution
        """
        eta = dist1.eta + dist2.eta
        return cls.from_eta(eta, dist1.fft)

    @classmethod
    def convolve(cls, dist1: Type['S1'], dist2: Type['S1']) -> Type['S1']:
        """
        Convolution of two distribution and update canonical parameters in log space for S1 group
        :param dist1: first distribution
        :param dist2: second distribution to convolve with
        :return: S1 distribution
        """
        M = dist1.M * dist2.M
        return cls.from_M(M, dist1.fft)

    def normalize(self):
        """
        Updated moments, energy, probability, log partition constant and compute Ms for S1 group
        :return: none
        """
        # Update moments and log partition function
        _, _ = self.compute_moments_lnz(self.eta, update=True)
        # Compute prob
        prob = np.exp(self.energy - self.l_n_z)
        # Compute Moments
        self.M = self.fft.analyze(prob)

    def compute_eta(self) -> None:
        """
        Compute eta from M, prob and energy for S1 group
        :return: none
        """
        # Compute energy
        energy = np.log(np.where(self.prob > 0, self.prob, 1e-8))
        # Compute eta
        self.eta = self.fft.analyze(energy)

    def compute_moments_lnz(self, eta: np.ndarray, update: bool = True) -> Tuple[np.ndarray, float]:
        """
        Adapted from https://github.com/AMLab-Amsterdam/lie_learn
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: moments and log partition constant
        """
        negative_energy = self.fft.synthesize(eta, oversample=True)
        maximum = np.max(negative_energy)
        unnormalized_moments = self.fft.analyze(np.exp(negative_energy - maximum), oversample=True)
        # Inverse shift FFT as default method shifts it
        unnormalized_moments = ifftshift(unnormalized_moments)
        # Scale by invariant haar measure
        unnormalized_moments[0] *= np.pi * 2
        # Assign moments and log partition function
        moments = unnormalized_moments[1:self.fft.bandwidth + 1] / unnormalized_moments[0]
        l_n_z = np.log(unnormalized_moments[0]) + maximum
        # Update moments of distribution and constant only when needed
        if update:
            self.moments = moments
            self.l_n_z = l_n_z.real
        return moments, l_n_z

    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """
        Energy of the distribution
        :param t: samples
        :return: energy of the distribution
        """
        raise NotImplementedError("This distribution does not have closed form to compute energy from samples, "
                                  "instead use `normalize` to obtain energy from eta")


class VonMises(S1):
    """
    Von Mises distribution for S1.
    Unlike a Gaussian, Von Mises produces a proper probability distribution.
    """

    def __init__(self, mu_theta: float = 0.0, kappa: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu_theta
        self.kappa = kappa
        self.from_samples()

    def log_prob(self, t: np.ndarray) -> np.ndarray:
        """Log probability of Von-mises distribution"""
        # p = np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))
        log_p = self.kappa * np.cos(t - self.mu) - np.log(2 * np.pi * i0(self.kappa))
        return log_p

    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """Energy Von-mises distribution"""
        energy = self.kappa * np.cos(t - self.mu)
        return energy


class S1Gaussian(S1):
    """
    A pseudo Gassian on S1.
    Needs to be renomalize to be a proper probability distribution.
    """

    def __init__(self, mu_theta: float = 0.0, cov: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mu_theta = mu_theta
        self.mu = self.theta_to_2D(np.array([mu_theta])).flatten()
        self.cov = cov
        self.from_samples()

    def compute_energy(self, theta):
        x = self.theta_to_2D(theta)
        angle = np.arccos(x.dot(self.mu))  # r = 1 for both so no denominator needed
        return -0.5 * np.power(angle, 2) / self.cov
        # return -1.0 / 2.0 * np.power(angle / self.cov, 2)

    def theta_to_2D(self, theta):
        r = 1.0
        out = np.empty(theta.shape + (2,))

        ct = np.cos(theta)
        st = np.sin(theta)
        out[..., 0] = r * ct
        out[..., 1] = r * st
        return out


class S1MultimodalGaussian(S1):
    """
    A mixture of Gaussians for s1.
    """

    def __init__(self, mu_list: List[float] = [0.0, 0.0], cov_list: List[float] = [0.5, 1.0], **kwargs):
        super().__init__(**kwargs)
        # List of distributions
        self.distributions = [S1Gaussian(mu_theta=mu, cov=cov, **kwargs) for mu, cov in zip(mu_list, cov_list)]
        # Compute number of components
        self.n_components = len(self.distributions)
        self.from_samples()

    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """Energy of multimodal Gaussian distribution"""
        energy = 0.0
        log_weight = np.log(1.0 / self.n_components)
        for dist in self.distributions:
            energy += np.exp(log_weight + dist.compute_energy(t))
        # Add small constant to avoid log(0)
        energy = np.log(energy + 1e-9)
        return energy


class StepS1(S1):
    """
    A step function on S1.
    """

    def __init__(self, up: float = np.pi, down: float = 2 * np.pi, scale: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.up = up
        self.down = down
        self.scale = scale
        self.from_samples()

    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """Step function for S1 group"""
        energy = np.logical_and(self.up < t, t < self.down) * self.scale
        return energy
