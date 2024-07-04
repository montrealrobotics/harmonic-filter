"""
Distributions defined over the SE(2) motion group.
"""
from typing import Tuple, Type, List
from abc import ABCMeta

import numpy as np
from scipy.stats import multivariate_normal

from src.distributions.distribution_base import HarmonicExponentialDistribution


class SE2(HarmonicExponentialDistribution, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO: Get rid of this once SE2 real fft is implemented.
    def from_samples(self) -> None:
        """
        Set up the distribution from samples
        :return : none
        """
        # Compute energy of samples
        energy = self.compute_energy(self.samples)
        _, _, _, _, _, self.eta = self.fft.analyze(energy)
        # This seems redundant, but as there is a loss of information in FFT analyze due to cartesian to polar
        # interpolation, the normalization constant is computed wrt to the energy synthesize by the eta params and not
        # by the one originally used as input. Therefore, normalizing the "original" energy starts giving bad results
        self.energy, _, _, _, _, _ = self.fft.synthesize(self.eta)

    @classmethod
    def product(cls, dist1: Type['SE2'], dist2: Type['SE2']) -> Type['SE2']:
        """
        Product of two distribution and update canonical parameters for S1 group
        :param dist1: first distribution
        :param dist2: second distribution to multiply with
        :return: S1 distribution
        """
        eta = dist1.eta + dist2.eta
        return cls.from_eta(eta, dist1.fft)

    @staticmethod
    def mul(fh1, fh2):

        assert fh1.shape == fh2.shape

        # The axes of fh are (r, p, q)
        # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
        # outside the range stored.
        # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
        # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
        p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
        q0 = fh1.shape[2] // 2

        # The lower and upper bound of the p-range
        a = p0 - q0
        b = int(p0 + np.ceil(fh2.shape[2] / 2.))

        # fh12 = np.zeros(fh1.shape, dtype=fh1.dtype)
        # for i in range(fh1.shape[0]):
        #     fh12[i, :, :] = fh1[i, :, :].dot(fh2[i, a:b, :])
        # One liner
        fh12 = np.einsum('rpn,rnn->rpn', fh1, fh2[:, a:b, :])
        # fh12 = np.c_[fh12]  #.transpose(2, 0, 1)
        return fh12

    @staticmethod
    def mulT(fh1, fh2):

        assert fh1.shape == fh2.shape

        # The axes of fh are (r, p, q) -> (p, n, m)
        # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
        # outside the range stored.
        # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
        # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
        p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
        q0 = fh1.shape[2] // 2

        # The lower and upper bound of the p-range
        a = p0 - q0
        b = int(p0 + np.ceil(fh2.shape[2] / 2.))

        fh12 = np.zeros(fh1.shape, dtype=fh1.dtype)
        for i in range(fh1.shape[0]):
            fh12[i, :, :] = fh1[i, :, :].dot(fh2[i, :, :].T)[:, a:b]

        # This is the right einops operation but it is super slow!
        # fh12_ = np.einsum('rpn,rqn->rpq', fh1, fh2)[:, :, a:b]

        # fh12 = np.c_[fh12]  #.transpose(2, 0, 1)
        return fh12

    @classmethod
    def convolve(cls, dist1: Type['SE2'], dist2: Type['SE2']) -> Type['SE2']:
        """
        Convolution of two distribution and update canonical parameters in log
        space for SE2 group
        :param dist1: first distribution
        :param dist2: second distribution to convolve with
        :return: SE2 distribution
        """
        M = cls.mulT(dist2.M, dist1.M)
        return cls.from_M(M, dist1.fft)

    def normalize(self) -> None:
        """
        Updated moments, energy, probability, log partition constant and compute
        Ms for SE2 group
        :return: none
        """
        # Update moments and log partition function
        _, _ = self.compute_moments_lnz(self.eta, update=True)
        # Compute prob - do not store as this is not the result of synthesize
        prob = np.exp(self.energy - self.l_n_z) + 1e-8
        # Compute Ms
        _, _, _, _, _, self.M = self.fft.analyze(prob)

    def compute_eta(self) -> None:
        """
        Compute eta from M, prob and energy for S1 group
        :return: none
        """
        # Compute energy
        energy = np.log(np.where(self.prob > 0, self.prob, 1e-8))
        # Compute eta
        _, _, _, _, _, self.eta = self.fft.analyze(energy)

    def compute_moments_lnz(self, eta: np.ndarray, update: bool = True) -> Tuple[np.ndarray, float]:
        """
        Adapted from https://github.com/AMLab-Amsterdam/lie_learn
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: moments and log partition constant
        """
        negative_energy, _, _, _, _, _ = self.fft.synthesize(eta)
        maximum = np.max(negative_energy)
        _, _, _, _, _, unnormalized_moments = self.fft.analyze(np.exp(negative_energy - maximum))
        # TODO: Figure out why z_0 is the 0 index.
        z_0 = 0
        z_1 = unnormalized_moments.shape[1] // 2
        z_2 = unnormalized_moments.shape[2] // 2
        # Scale by invariant haar measure
        # Haar measure in Chirikjian's book
        # unnormalized_moments[z_0, z_1, z_2] *= np.power(2 * np.pi, 2)
        # Haar measure in Chirikjian's book for S1, semi-direct product R^2 \cross S1
        unnormalized_moments[z_0, z_1, z_2] *= np.pi * 2
        moments = unnormalized_moments / unnormalized_moments[z_0, z_1, z_2]
        moments[z_0, z_1, z_2] = unnormalized_moments[z_0, z_1, z_2]
        l_n_z = np.log(unnormalized_moments[z_0, z_1, z_2]) + maximum
        # Update moments of distribution and constant only when needed
        if update:
            self.moments = moments
            self.l_n_z = l_n_z.real
        return moments, l_n_z.real

    def compute_energy(self, t: np.ndarray) -> np.ndarray:
        """
        Energy of the distribution
        :param t: samples
        :return: energy of the distribution
        """
        raise NotImplementedError("This distribution does not have closed form to compute energy from samples, "
                                  "instead use `normalize` to obtain energy from eta")

    def update_params(self) -> None:
        """
        Update parameters of the distribution
        :return: none
        """
        raise NotImplementedError("This class does not have natural parameters thus they cannot be updated")

    @property
    def energy(self) -> np.ndarray:
        if self._energy is None:
            self._energy, _, _, _, _, _ = self.fft.synthesize(self.eta)
        return self._energy.real

    @energy.setter
    def energy(self, energy: np.ndarray) -> None:
        self._energy = energy.copy()

    @property
    def prob(self) -> np.ndarray:
        if self._prob is None:
            self._prob, _, _, _, _, _ = self.fft.synthesize(self.M)
            self._prob = np.where(self._prob.real > 0, self._prob.real, 1e-8)
        return self._prob

    @prob.setter
    def prob(self, prob: np.ndarray) -> None:
        self._prob = prob.copy()


class SE2Gaussian(SE2):
    """ Class to represent Gaussian-Like distributions in SE2. """

    def __init__(self,
                 mu: np.ndarray = np.zeros(3),
                 cov: np.ndarray = np.eye(3),
                 **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.from_samples()

    def update_params(self) -> None:
        raise NotImplementedError

    def compute_energy(self, x):
        assert x.shape[1] == 3
        diff = x - self.mu
        # Wrap angle
        diff[:, 2] = (diff[:, 2] + np.pi) % (2 * np.pi) - np.pi
        logpdf = multivariate_normal.logpdf(diff, mean=np.zeros(3), cov=self.cov)

        return logpdf.reshape(self.fft.spatial_grid_size)


class SE2MultimodalGaussian(SE2):
    """
    Class to represent a mixture of Gaussians for SE(2).
    """

    def __init__(self, mu_list: List[np.ndarray] = [np.zeros(3), np.zeros(3)],
                 cov_list: List[np.ndarray] = [np.eye(3), np.eye(3) * 0.1],
                 **kwargs):
        super().__init__(**kwargs)
        # List of distributions
        self.distributions = [SE2Gaussian(mu=mu, cov=cov, **kwargs) for mu, cov in zip(mu_list, cov_list)]
        # Compute number of components
        self.n_components = len(self.distributions)
        self.from_samples()

    def compute_energy(self, x):
        energy = 0.0
        log_weight = np.log(1.0 / self.n_components)
        for dist in self.distributions:
            energy += np.exp(log_weight + dist.compute_energy(x))
        # Add small constant to avoid log(0)
        energy = np.log(energy + 1e-9)
        return energy


class SE2Square(SE2):
    def __init__(self, x_limits: List[float], y_limits: List[float], theta_limits: List[float], scale: float, **kwargs):
        super().__init__(**kwargs)
        # Bounds for the square
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.theta_limits = theta_limits
        self.scale = scale
        self.from_samples()

    def compute_energy(self, x):
        """Square function in XY for SE2"""
        x_energy = np.logical_and(self.x_limits[0] < x[:, 0], x[:, 0] < self.x_limits[1])
        y_energy = np.logical_and(self.y_limits[0] < x[:, 1], x[:, 1] < self.y_limits[1])

        diff_t = (x[:, 2] + np.pi) % (2 * np.pi) - np.pi
        #print(diff_t)
        t_energy = np.logical_and(self.theta_limits[0] < diff_t, diff_t < self.theta_limits[1])
        energy = np.logical_and(np.logical_and(x_energy, y_energy), t_energy) * self.scale
        return energy.reshape(self.fft.spatial_grid_size)
