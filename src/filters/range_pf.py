"""
A class that implements a particle filter.
"""
from typing import Tuple

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from src.utils.door_dataset_utils import preprocess_mask

from einops import rearrange


class RangePF:
    def __init__(self,
                 prior: np.ndarray,
                 prior_cov: np.ndarray,
                 n_particles: int = 100):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        :param n_particles: Number of particles to use.
        """
        self._N = n_particles
        self.particles = (np.linalg.cholesky(prior_cov) @ np.random.randn(3, self._N)).T + prior
        self.weights = np.ones(self._N) / self._N
        self.mode_index = None

    def prediction(self,
                   step: np.ndarray,
                   step_cov: np.ndarray) -> None:
        """
        Prediction step PF
        :param step: motion step (relative displacement) of dimension (3,)
        :param step_cov: Covariance matrix of prediction step of dimension (3, 3)
        :return Mean and covariance of belief
        """
        # sample steps
        step = (np.linalg.cholesky(step_cov) @ np.random.randn(3, self._N)).T + step
        step[:, 2] = (step[:, 2] + np.pi) % (2 * np.pi) - np.pi
        # Apply step
        c = np.cos(self.particles[:, 2])
        s = np.sin(self.particles[:, 2])
        self.particles[:, 0] += c * step[:, 0] - s * step[:, 1]
        self.particles[:, 1] += s * step[:, 0] + c * step[:, 1]
        self.particles[:, 2] += step[:, 2]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update(self,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> np.ndarray:
        """
        Update step PF
        :param landmarks: location of each UWB landmark in the map (n, 3)
        :param observations: range measurements of dimension (n,)
        :param observations_cov: variance of each measurement of dimension (n,)
        :return Mean of the particles
        """
        observations_std = np.sqrt(observations_cov)
        ### Not independent measurements ###
        # weight = 1 / len(observations_std)
        # for i, landmark in enumerate(landmarks):
        #     dist = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1)
        #     self.weights += norm(dist, observations_std[i]).pdf(observations[i]) * weight
        ### Independence between measurements ###
        self.weights = np.log(self.weights + 1e-8)
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1)
            prob = norm(distance, observations_std[i]).pdf(observations[i]) + 1e-8
            self.weights += np.log(prob)

        # Normalize weights
        self.weights -= logsumexp(self.weights)
        self.weights = np.exp(self.weights)

        ## Resample
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        # Low variance sampling
        r = np.random.rand() / self._N
        samples = np.linspace(0.0, 1.0, num=self._N, endpoint=False) + r
        indexes = np.searchsorted(cumulative_sum, samples)

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.mode_index = self.weights.argmax()
        self.weights.fill(1.0 / self._N)

        return np.mean(self.particles, axis=0)

    def compute_mode(self) -> np.ndarray:
        """
        Compute mode of the distribution
        :return mode of distribution
        """
        return self.particles[self.mode_index]

    def neg_log_likelihood(self, pose: np.ndarray, grid_bounds: Tuple[float], grid_size: Tuple[float]) -> np.ndarray:
        """
        Evaluate posterior distribution of histogram filter
        :param pose: Pose at which to interpolate the SE2 Fourier transform
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        :return ll: Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        x_bins = np.linspace(grid_bounds[0], grid_bounds[1], num=grid_size[0] + 1, endpoint=True)  # Define x bins
        y_bins = np.linspace(grid_bounds[0], grid_bounds[1], num=grid_size[1] + 1, endpoint=True)  # Define y bins
        theta_bins = np.linspace(-np.pi, np.pi, num=grid_size[2] + 1, endpoint=True)  # Define theta
        # Compute density defined by particles
        density, _ = np.histogramdd(self.particles,
                                    bins=(x_bins.flatten(), y_bins.flatten(), theta_bins.flatten()),
                                    density=True)
        # Find the closest bin to current pose
        x_index = np.digitize(pose[0], x_bins[:-1], right=True) - 1
        y_index = np.digitize(pose[1], y_bins[:-1], right=True) - 1
        theta_index = np.digitize(pose[2], theta_bins[:-1], right=True) - 1
        # Get density and if NaN, set to zero
        p_g = 0.0 if np.isnan(density[x_index, y_index, theta_index]) else density[x_index, y_index, theta_index]
        # Prevent zero likelihood error
        ll = np.log(p_g + 1e-8)
        return -ll


class BearingPF(RangePF):
    def __init__(self, d_door2pose: float = 0.1, **kwargs):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        :param n_particles: Number of particles to use.
        """
        super().__init__(**kwargs)
        self.d_door2pose = d_door2pose

    def update(self,
               landmarks: np.ndarray,
               map_mask: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> np.ndarray:
        """
        Update step PF for bearing-only measurements
        :param landmarks: location of all doors in the map (x-y coordinates) (n, 2)
        :param map_mask: Binary mask indicating traversable area, computed from cost map
        :param observations: bearing measurements of dimension (m,)
        :param observations_cov: variance of each door of dimension (n,)
        :return Mean of the particles
        """
        observations_std = np.sqrt(observations_cov)
        # Map weights temporarily to log space
        self.weights = np.log(self.weights + 1e-9)
        ### independent measurements but p(z_{t,i} | x_t, m) is a mixture of n_doors components ###
        for i, obs in enumerate(observations):
            diff =  rearrange(landmarks, "n m -> n 1 m") - rearrange(self.particles[:, :2], "p m -> 1 p m")
            angle = np.arctan2(diff[:, :, 1], diff[:, :, 0]) - self.particles[:, 2]
            # Wrap angle.
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            diff_angle = obs - angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            # This mask will blackout particles outside map
            mask = preprocess_mask(map_mask, self.particles)
            mixture = mask * norm(diff_angle, rearrange(observations_std, "n -> n 1")).pdf(0.0)
            mixture = mixture.max(0) + 1e-8
            # max along components dimension
            self.weights = np.maximum(self.weights, np.log(mixture))

        # Normalize weights
        self.weights = np.where(mask == 0, -np.inf, self.weights) # Mask out particles outside map
        self.weights -= logsumexp(self.weights)
        self.weights = np.exp(self.weights)

        ## Resample
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        # Low variance sampling
        r = np.random.rand() / self._N
        samples = np.linspace(0.0, 1.0, num=self._N, endpoint=False) + r
        indexes = np.searchsorted(cumulative_sum, samples)

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.mode_index = self.weights.argmax()
        self.weights.fill(1.0 / self._N)

        return np.mean(self.particles, axis=0)