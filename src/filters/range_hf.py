"""
A python class implementing a range-only histogram filter
"""
from typing import Tuple, List

import numpy as np
from einops import rearrange
from scipy.stats import multivariate_normal, norm
from scipy.ndimage.filters import gaussian_filter
from scipy.special import logsumexp

import matplotlib.pyplot as plt


class RangeHF:
    def __init__(self,
                 prior: np.ndarray,
                 prior_cov: np.ndarray,
                 grid_samples: np.ndarray,
                 grid_bounds: Tuple[float] = (-0.5, 0.5),
                 grid_size: Tuple[int] = (50, 50, 32)):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: covariance noise for the prior distribution of dimension (3, 3)
        :param grid_samples: samples of the grid (x, y, theta)
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        """
        self.grid_samples: np.ndarray = grid_samples
        self.grid_bounds: Tuple[float] = grid_bounds
        self.grid_size: Tuple[int] = grid_size
        self._N = np.prod([ax for ax in grid_size])
        self.step_xy = (grid_bounds[1] - grid_bounds[0]) / grid_size[0]
        self.step_theta = (np.pi * 2) / grid_size[2]
        self.volume = self.step_xy * self.step_xy * self.step_theta
        # Define prior and normalize it
        self.prior = multivariate_normal(prior, prior_cov).pdf(self.grid_samples)
        self.prior /= np.sum(self.prior)
        # self.plot(self.prior[:].reshape(self.grid_size), "Posteriori")

    def prediction(self,
                   step: np.ndarray,
                   step_cov: np.ndarray) -> None:
        """
        Prediction step HF
        :param step: motion step (relative displacement) of dimension (3,)
        :param step_cov: covariance matrix of prediction step of dimension (3, 3)
        :return none
        """
        # Apply step to obtain expected transition
        centroids = self.grid_samples.copy()
        c = np.cos(centroids[:, 2])
        s = np.sin(centroids[:, 2])
        centroids[:, 0] += c * step[0] - s * step[1]
        centroids[:, 1] += s * step[0] + c * step[1]
        centroids[:, 2] += step[2]
        centroids[:, 2] = (centroids[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Compute indices of new centroids
        bins = self._compute_bin_indices(centroids)
        # Filter x-y coordiantes that went off the grid
        mask = self._filter_out_of_bounds(centroids)
        # Update prior belief
        belief = np.zeros_like(self.prior).reshape(self.grid_size)
        np.add.at(belief, tuple(bins.T), self.prior * mask)
        # Add noise according to the process model's noise - implemented as a Gaussian blur
        scaled_sigma = np.sqrt(np.diag(step_cov.copy())) / np.array([self.step_xy, self.step_xy, self.step_theta])
        belief = gaussian_filter(belief, scaled_sigma, mode="constant")
        # Normalize belief
        belief = belief / np.sum(belief)
        # If there is no probability mass over new belief, just return previous one
        if np.sum(belief) != 0.:
            self.prior = belief.flatten()

    def update(self,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> np.ndarray:
        """
        Update step HF
        :param landmarks: location of each UWB landmark in the map (n, 3)
        :param observations: range measurements of dimension (n,)
        :param observations_cov: variance of each measurement of dimension (n,)
        :return Mean of the particles
        """
        observations_std = np.sqrt(observations_cov)
        ### Not independent measurements ###
        # weight = 1 / len(observations_std)
        # measurement_likelihood = 1e-300
        # for i, landmark in enumerate(landmarks):
        #     dist = np.linalg.norm(landmark - self.grid_samples[:, :2], axis=1)
        #     measurement_likelihood += norm(dist, observations_std[i]).pdf(observations[i]) * weight
        ### Independence between measurements ###
        measurement_likelihood = 0.
        for i, landmark in enumerate(landmarks):
            dist = np.linalg.norm(landmark - self.grid_samples[:, :2], axis=1)
            prob = norm(dist, observations_std[i]).pdf(observations[i]) + 1e-8
            measurement_likelihood += np.log(prob)

        measurement_likelihood -= logsumexp(measurement_likelihood)
        if measurement_likelihood is not None:
            # Combine the prior belief and the measurement likelihood to get the posterior belief
            p_belief = self.prior * np.exp(measurement_likelihood)
            # Normalizing the posterior belief
            if np.sum(p_belief) != 0.:
                self.prior = p_belief / np.sum(p_belief)

        # Compute mean of histogram filter
        mean = self._compute_mean()
        return mean

    def compute_mode(self) -> np.ndarray:
        """
        Compute mode of the distribution
        :return mode of distribution
        """
        return self.grid_samples[self.prior.argmax()]

    def _filter_out_of_bounds(self, centroids: np.ndarray) -> np.ndarray:
        """
        Filter out of bounds centroids
        :param centroids: propagated centroids after motion step
        :return boolean mask with in-bound centroids
        """
        mask_x = (centroids[..., 0] >= self.grid_bounds[0]) & (centroids[..., 0] <= self.grid_bounds[1])
        mask_y = (centroids[..., 1] >= self.grid_bounds[0]) & (centroids[..., 1] <= self.grid_bounds[1])
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def _compute_bin_indices(self, centroids: np.ndarray) -> np.ndarray:
        """
        Compute bin indices for each centroid in the grid
        :param centroids: propagated centroids after motion step
        :return bin index of each centroid
        """
        idx = np.digitize(centroids[:, 0], right=False,
                          bins=np.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[0], endpoint=False))
        idy = np.digitize(centroids[:, 1], right=False,
                          bins=np.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[1], endpoint=False))
        # Transform angle from [-pi, pi] to [0, 2pi] as the former is the actual specification of the grid
        angles = centroids[:, 2].copy() % (2 * np.pi)
        idt = np.digitize(angles, right=False, bins=np.linspace(0, 2 * np.pi, self.grid_size[2], endpoint=False))

        return np.stack([idx, idy, idt], axis=-1) - 1

    def _compute_mean(self) -> np.ndarray:
        """
        Compute expected value of the histogram bins
        :return mean value of the histogram bins
        """
        prod = self.grid_samples * rearrange(self.prior, 'n -> n 1')
        mean = np.sum(prod, axis=0)
        return mean

    def plot(self, belief: np.ndarray, title: str = "") -> None:
        # Plotting routine for debugging
        xs = np.linspace(-0.5, 0.5, self.grid_size[0], endpoint=False)
        ys = np.linspace(-0.5, 0.5, self.grid_size[1], endpoint=False)
        x, y = np.meshgrid(xs, ys, indexing='ij')
        h = plt.contourf(x, y, belief.sum(-1))
        plt.axis('scaled')
        plt.colorbar()
        plt.show()

    def neg_log_likelihood(self, pose) -> np.ndarray:
        """
        Evaluate posterior distribution of histogram filter
        :param pose: Pose at which to interpolate the SE2 Fourier transform
        :return ll: Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        # Grid samples are between [0, 2pi] so we need to transform the pose to that range
        wrapped_pose = pose.copy()
        wrapped_pose[2] = wrapped_pose[2] % (2 * np.pi)
        idx = np.argmin(np.linalg.norm(self.grid_samples - wrapped_pose, axis=-1))
        # Divide by cube's volume to obtain pdf
        ll = np.log((self.prior[idx] / self.volume) + 1e-8)
        return -ll


class RangeHFBimodal(RangeHF):
    def __init__(self,
                 priors: np.ndarray,
                 priors_cov: np.ndarray,
                 grid_samples: np.ndarray,
                 grid_bounds: Tuple[float] = (-0.5, 0.5),
                 grid_size: Tuple[int] = (50, 50, 32)):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: covariance noise for the prior distribution of dimension (3, 3)
        :param grid_samples: samples of the grid (x, y, theta)
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        """
        self.grid_samples: np.ndarray = grid_samples
        self.grid_bounds: Tuple[float] = grid_bounds
        self.grid_size: Tuple[int] = grid_size
        # Assume the two modes are equally likely
        self.pi = np.array([0.5, 0.5])
        self._N = np.prod([ax for ax in grid_size])
        self.step_xy = (grid_bounds[1] - grid_bounds[0]) / grid_size[0]
        self.step_theta = (np.pi * 2) / grid_size[2]
        self.volume = self.step_xy * self.step_xy * self.step_theta
        # Define bimodal prior and normalize it
        self.prior = self.pi[0] * multivariate_normal(priors[0], priors_cov[0]).pdf(self.grid_samples) + self.pi[1] * multivariate_normal(priors[1], priors_cov[1]).pdf(self.grid_samples)
        self.prior /= np.sum(self.prior)
        # self.plot(self.prior[:].reshape(self.grid_size), "Posteriori")
    def prediction(self, step, step_cov):
        return super().prediction(step, step_cov)
    
    def update(self, landmarks, observations, observations_cov):
        return super().update(landmarks, observations, observations_cov)
    

class BearingHF(RangeHF):
    def __init__(self, d_door2pose: float = 0.1, **kwargs):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: covariance noise for the prior distribution of dimension (3, 3)
        :param grid_samples: samples of the grid (x, y, theta)
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        """
        super().__init__(**kwargs)
        self.d_door2pose = d_door2pose

    def update(self,
               landmarks: np.ndarray,
               map_mask: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> np.ndarray:
        """
        Update step HF
        :param landmarks: location of each UWB landmark in the map (n, 2)
        :param map_mask: Binary mask indicating traversable area, computed from cost map
        :param observations: range measurements of dimension (m,)
        :param observations_cov: variance of each door of dimension (n,)
        :return Mean of the particles
        """
        observations_std = np.sqrt(observations_cov)
        ### independent measurements but p(z_{t,i} | x_t, m) is a mixture of n_doors components ###
        measurement_likelihood = np.log(1e-9)
        for i, obs in enumerate(observations):
            diff = rearrange(landmarks, "n m -> n 1 m") - rearrange(self.grid_samples[:, :2], "p m -> 1 p m")
            angle = np.arctan2(diff[:, :, 1], diff[:, :, 0]) - ((self.grid_samples[:, 2] + np.pi) % (2 * np.pi) - np.pi)
            # Wrap angle.
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            diff_angle = obs - angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            mixture = map_mask * norm(diff_angle, rearrange(observations_std, "n -> n 1")).pdf(0.0)
            mixture = mixture.max(0) + 1e-8
            # max along components dimension
            measurement_likelihood = np.maximum(measurement_likelihood, np.log(mixture))

        measurement_likelihood -= logsumexp(measurement_likelihood)
        if measurement_likelihood is not None:
            # Combine the prior belief and the measurement likelihood to get the posterior belief
            p_belief = self.prior * np.exp(measurement_likelihood)
            # Normalizing the posterior belief
            if np.sum(p_belief) != 0.:
                self.prior = p_belief / np.sum(p_belief)

        # Compute mean of histogram filter
        mean = self._compute_mean()
        return mean
