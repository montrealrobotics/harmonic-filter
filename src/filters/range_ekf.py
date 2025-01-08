"""
A class for a harmonic bayesian filter
"""
from typing import Tuple, List


import numpy as np
from scipy.stats import multivariate_normal
from src.groups.se2_group import SE2Group
from scipy.special import logsumexp

from einops import rearrange


class RangeEKF:
    def __init__(self,
                 prior: np.ndarray,
                 prior_cov: np.ndarray):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        """
        self.pose: SE2Group = SE2Group.from_parameters(*prior)
        self.state_cov: np.ndarray = prior_cov.copy()

    def prediction(self,
                   step: np.ndarray,
                   step_cov: np.ndarray) -> None:
        """
        Prediction step EKF
        :param step: motion step (relative displacement) of dimension (3,)
        :param step_cov: Covariance matrix of prediction step of dimension (3, 3)
        :return None
        """
        # Get group representation of current pose
        pose = self.pose.parameters()
        # Compute prediction jacobian
        g = np.array([[1, 0, -step[0] * np.sin(pose[2]) - step[1] * np.cos(pose[2])],
                      [0, 1, step[0] * np.cos(pose[2]) - step[1] * np.sin(pose[2])],
                      [0, 0, 1]])
        # Matrix representation of current pose
        step = SE2Group.from_parameters(step[0], step[1], step[2])
        # Propagate step (mean)
        self.pose = self.pose @ step
        # Update covariance of the state
        self.state_cov = g @ self.state_cov @ g.T + step_cov

        return None

    def update(self,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step EKF
        :param landmarks: location of each UWB landmark in the map (n, 3)
        :param observations: range measurements of dimension (n,)
        :param observations_cov: variance of each measurement of dimension (n,)
        :return normalized belief as the mean and covariance of a gaussian distribution
        """
        # Compute mean measurements
        q = landmarks - self.pose.parameters()[:2]
        z_hat = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
        h = np.asarray([-q[:, 0] / z_hat, -q[:, 1] / z_hat, np.zeros_like(z_hat)]).T
        # Update
        s = h @ self.state_cov @ h.T + np.diag(observations_cov)
        # Compute Kalman gain
        k = self.state_cov @ h.T @ np.linalg.inv(s)
        # Update state and state covariance
        pose = self.pose.parameters() + k @ (observations - z_hat)
        pose[2] = (pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.pose = SE2Group.from_parameters(*pose)
        self.state_cov = (np.eye(3) - k @ h) @ self.state_cov

        return self.pose.parameters(), self.state_cov

    def neg_log_likelihood(self, pose) -> np.ndarray:
        """
        Evaluate posterior distribution of a multivariate gaussian
        :param pose: Pose at which to interpolate the SE2 Fourier transform
        :return ll: Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        ll = multivariate_normal.logpdf(pose, mean=self.pose.parameters(), cov=self.state_cov)
        return -ll


class RangeEKFBimodal(RangeEKF):
    def __init__(self,
                 priors: List[np.ndarray],
                 priors_cov: List[np.ndarray], 
                 grid_size: Tuple[int] = (50, 50, 32)):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        """
        self.grid_size: Tuple[int] = grid_size
        # Initialize the two modes
        pose1: SE2Group = SE2Group.from_parameters(*priors[0])
        pose2: SE2Group = SE2Group.from_parameters(*priors[1])
        self.pose = [pose1, pose2]
        # Assume the two modes are equally likely
        self.pi = np.array([0.5, 0.5])
        cov1: np.ndarray = priors_cov[0].copy()
        cov2: np.ndarray = priors_cov[1].copy()
        self.state_cov = [cov1, cov2]
        self.mixture_mean = np.zeros(3)
        self.mixture_mode = np.zeros(3)
        self.mixture_cov = np.zeros((3, 3))
        self.compute_mixture_statistics()
        self.pose = SE2Group.from_parameters(self.mixture_mean[0], self.mixture_mean[1], self.mixture_mean[2])
        self.state_cov = self.mixture_cov


    def prediction(self,
                   step: np.ndarray,
                   step_cov: np.ndarray) -> None:
        """
        Prediction step EKF
        :param step: motion step (relative displacement) of dimension (3,)
        :param step_cov: Covariance matrix of prediction step of dimension (3, 3)
        :return None
        """
        return super().prediction(step, step_cov)

    def update(self,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step EKF
        :param landmarks: location of each UWB landmark in the map (n, 3)
        :param observations: range measurements of dimension (n,)
        :param observations_cov: variance of each measurement of dimension (n,)
        :return normalized belief as the mean and covariance of a gaussian distribution
        """
        return super().update(landmarks, observations, observations_cov)
    
    def compute_mixture_statistics(self):
        """
        Compute the mean and covariance of the mixture distribution
        """
        self.mixture_mean = self.pi[0] * self.pose[0].parameters() + self.pi[1] * self.pose[1].parameters()
        self.mixture_cov = np.zeros((3, 3))
        for p in range(2):
            self.mixture_cov += self.pi[p] * (self.state_cov[p] + (self.pose[p].parameters() - self.mixture_mean) @ (self.pose[p].parameters() - self.mixture_mean).T)
        self.mixture_mode = self.pose[np.argmax(self.pi)].parameters()

    def approximate_grid_density(self):
        """
        Approximate the density of the mixture distribution using a grid
        """
        xs = np.linspace(-0.5, 0.5, self.grid_size[0], endpoint=False)
        ys = np.linspace(-0.5, 0.5, self.grid_size[1], endpoint=False)
        x, y = np.meshgrid(xs, ys, indexing='ij')
        pos = np.dstack((x, y))
        # Appoximate density in the grid
        rv1 = multivariate_normal(self.pose[0].parameters()[:2], self.state_cov[0][:2, :2])
        rv2 = multivariate_normal(self.pose[1].parameters()[:2], self.state_cov[1][:2, :2])
        z1 = rv1.pdf(pos)
        z2 = rv2.pdf(pos)
        z = self.pi[0] * z1 + self.pi[1] * z2
        return x, y, z

    def neg_log_likelihood(self, pose):
        """
        Evaluate posterior distribution of a multivariate gaussian
        :param pose: Pose at which to interpolate the SE2 Fourier transform
        :return ll: Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        return super().neg_log_likelihood(pose)
    

class BearingEKF(RangeEKF):
    def __init__(self, **kwargs):
        """
        :param prior: a prior pose of as a numpy array of dimension (3,)
        :param prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        """
        super().__init__(**kwargs)

    def update(self,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step EKF
        :param landmarks: location of each UWB landmark in the map (n, 2)
        :param observations: range measurements of dimension (m,)
        :param observations_cov: variance of each door of dimension (n,)
        :return normalized belief as the mean and covariance of a gaussian distribution
        """
        # Find the nearest landmarks to current measurements
        diff = landmarks - self.pose.parameters()[:2]
        angle = np.arctan2(diff[:, 1], diff[:, 0]) - self.pose.parameters()[2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        # Find nearest neighbor landmark for each observation
        indices = np.argmin(np.abs(rearrange(angle, "n -> 1 n") - rearrange(observations, "m -> m 1")), axis=-1)
        # Obtain mean measurement for each observation
        z_hat = angle[indices]
        # Compute measurement Jacobian
        q_diff = diff[indices]
        q = q_diff[:, 0] ** 2 + q_diff[:, 1] ** 2
        h = np.asarray([q_diff[:, 1] / q, -q_diff[:, 0] / q, -np.ones_like(z_hat)]).T
        # Update
        s = h @ self.state_cov @ h.T + np.diag(observations_cov[indices])
        # Compute Kalman gain
        k = self.state_cov @ h.T @ np.linalg.inv(s)
        innovation = (observations - z_hat + np.pi) % (2 * np.pi) - np.pi
        # Update state and state covariance
        pose = self.pose.parameters() + k @ innovation
        pose[2] = (pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.pose = SE2Group.from_parameters(*pose)
        self.state_cov = (np.eye(3) - k @ h) @ self.state_cov

        return self.pose.parameters(), self.state_cov
