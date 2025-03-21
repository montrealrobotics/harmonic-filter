"""
A class for a harmonic bayesian filter
"""
from typing import Type, List
from copy import deepcopy
import numpy as np
from numpy.fft import ifftshift
from scipy.interpolate import RegularGridInterpolator
from einops import rearrange

from src.distributions.distribution_base import HarmonicExponentialDistribution
from src.spectral.se2_fft import SE2_FFT


class BayesFilter:
    def __init__(
        self,
        distribution: Type[HarmonicExponentialDistribution],
        prior: HarmonicExponentialDistribution,
    ):
        """
        :param distribution: Type of distribution to filter.
        :param prior: a prior distribution of type "distribution".
        """
        self.distribution = distribution
        self.prior: HarmonicExponentialDistribution = prior

    def prediction(
        self, motion_model: HarmonicExponentialDistribution
    ) -> HarmonicExponentialDistribution:
        """
        Prediction step
        :param motion_model: motion model for prediction step
        :return unnormalized belief distribution
        """
        # Convolve prior and motion model
        predict = self.distribution.convolve(motion_model, self.prior)
        # Update prior
        self.prior = deepcopy(predict)

        return predict

    def update(
        self, measurement_model: HarmonicExponentialDistribution
    ) -> HarmonicExponentialDistribution:
        """
        Update step
        :param measurement_model: measurement model for update step
        :return unnormalized posterior distribution
        """
        # Product of belief and measurement model
        update = self.distribution.product(self.prior, measurement_model)
        update.normalize()
        # Update prior with new belief
        self.prior = deepcopy(update)

        return update, measurement_model

    @staticmethod
    def neg_log_likelihood(
        eta: np.ndarray, l_n_z: float, pose: np.ndarray, se2_fft: SE2_FFT
    ) -> float:
        """
        Compute point-wise synthesize the SE2 Fourier transform M at a given pose. More explicitly, this function
        computes p(g = pose)
        Args:
            eta (np.array): Fourier coefficients (eta) of SE2 distribution with shape [n, 3] where n is the number of
                            samples
            l_n_z (float): Log of normalization constant of SE2 distribution
            pose (np.array): Pose at which to interpolate the SE2 Fourier transform
            se2_fft (SE2_FFT): Object class for SE2 Fourier transform

        Returns:
            Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        # Reshape in case single pose is provided
        if pose.ndim < 2:
            pose = rearrange(pose, "b -> 1 b")
        # Arrange pose samples in broadcastable shape
        dx, dy = rearrange(pose[:, 0], "b -> 1 b"), rearrange(pose[:, 1], "b -> 1 b")
        d_theta = rearrange(pose[:, 2], "b -> 1 b")
        # Synthesize signal to obtain first FFT and
        _, _, _, f_p_psi_m, _, _ = se2_fft.synthesize(eta)
        # Shift the signal to the origin
        f_p_psi_m = rearrange(ifftshift(f_p_psi_m, axes=2), "p n m -> p n m 1")
        # Theta ranges from 0 to 2pi, thus ts = 2 * np.pi (duration)
        t_theta = 2 * np.pi
        n_theta = f_p_psi_m.shape[2]
        # Evaluate fourier coefficients at desired point
        omega_n = (
            2 * np.pi * (1 / t_theta) * rearrange(np.arange(n_theta), "n -> 1 1 n 1")
        )
        # Compute the value of f(x) using the inverse Fourier transform
        f_p_psi = np.sum(f_p_psi_m * np.exp(-1j * omega_n * d_theta), axis=2)
        # f_p_psi = np.sum(f_p_psi_m * np.exp(-1j * omega_n * d_theta / (2. * np.pi)), axis=2)
        # Map from polar to cartesian grid
        f_p_p = se2_fft.resample_p2c_3d(f_p_psi)
        # Finally, 2D inverse FFT
        f_p_p = ifftshift(f_p_p, axes=(0, 1))
        # Set domain of X and Y, recall X and Y range from [-0.5, 0.5]
        t_x, t_y = 1.0, 1.0
        n_x, n_y = f_p_p.shape[:2]
        # Compute complex term
        angle_x = (
            1j * 2 * np.pi * (1 / t_x) * rearrange(np.arange(n_x), "nx -> nx 1") * dx
        )  # Angle component in X
        angle_y = (
            1j * 2 * np.pi * (1 / t_y) * rearrange(np.arange(n_y), "ny -> ny 1") * dy
        )  # Angle component in Y
        angle = rearrange(angle_x, "nx b -> nx 1 b") + rearrange(
            angle_y, "ny b -> 1 ny b"
        )
        # Compute the value of log(p(g)) using the inverse Fourier transform
        f = np.sum(f_p_p * np.exp(angle), axis=(0, 1)).real - l_n_z
        return -f

    @staticmethod
    def neg_log_likelihood2(
        energy: np.ndarray, l_n_z: float, pose: np.ndarray, size: List
    ) -> float:
        xs = np.linspace(-0.5, 0.5, size[0], endpoint=False)
        ys = np.linspace(-0.5, 0.5, size[1], endpoint=False)
        ts = np.linspace(0.0, 2.0 * np.pi, size[2], endpoint=False)
        gt_pose = pose
        # Reshape in case single pose is provided
        if gt_pose.ndim < 2:
            gt_pose = rearrange(pose, "b -> 1 b")
        gt_pose[:, 2] = gt_pose[:, 2] % (2.0 * np.pi)
        gt_pose[:, 2] = min(ts[-1], gt_pose[:, 2])
        interpolator = RegularGridInterpolator((xs, ys, ts), energy)
        e = interpolator(gt_pose)
        return -e + l_n_z
