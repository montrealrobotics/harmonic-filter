from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from src.distributions.se2_distributions import SE2, SE2Gaussian
from src.simulators.simulator_base import Simulator
from src.spectral.base_fft import FFTBase
from src.groups.se2_group import SE2Group
from src.utils.logging import get_logger

log = get_logger(__name__)


class SE2RangeSimulator(Simulator):
    def __init__(self, start: SE2Group = SE2Group(),
                 step: SE2Group = SE2Group.from_parameters(0.1, 0.05, np.pi / 4),
                 samples: Optional[np.ndarray] = None,
                 fft: FFTBase = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.position = start
        self.step = step
        self.beacons = np.array(
            [[0, 0.1],
             [0, 0.05],
             [0, 0.0],
             [0, -0.05],
             [0, -0.1]])
        self.beacon_idx = 0
        self.samples = samples
        self.fft = fft
        # Container
        self.range_measurement: Optional[np.ndarray] = None
        # If motion noise or measurement noise are zero, default to 1e-4 and 5e-3 respectively
        self.motion_cov = np.diag(self.motion_noise ** 2) if self.motion_noise.sum() != 0.0 else np.eye(3) * 1e-4
        self.measurement_cov = self.measurement_noise ** 2 if self.measurement_noise != 0.0 else 1e-3

    def motion(self) -> SE2:
        """
        Simulate a motion with step.
        :return: SE2 distribution of relative predicted motion.
        """
        # self.position = self.step @ self.position
        self.position = self.position @ self.step
        log.info(f"Motion step (x, y, theta): {self.position.parameters()}")
        # Jitter step with noise and wrap heading between 0 and 2pi
        noisy_prediction = self.step.parameters() + np.random.randn(3) * self.motion_noise
        noisy_prediction[2] = noisy_prediction[2] % (2 * np.pi)

        return SE2Gaussian(noisy_prediction,
                           self.motion_cov,
                           samples=self.samples,
                           fft=self.fft)

    def measurement(self) -> SE2:
        """
        Simulate measurement
        :return: current position as the measurement as a vector of [x, y, theta].
        """
        self._update_beacon_idx()
        range_beacon = self.beacons[self.beacon_idx, :]
        # Observation z_t
        self.range_measurement = np.linalg.norm(self.position.parameters()[0:2] - range_beacon)
        # Jitter range measurement with noise
        self.range_measurement += np.random.normal(0.0, self.measurement_noise, 1).item()
        dist = np.linalg.norm(range_beacon - self.samples[:, 0:2], axis=1)
        ### Log prob function ###
        range_prob = norm(dist, self.measurement_noise).pdf(self.range_measurement)
        range_ll = np.log(range_prob + 1e-8)
        ### Energy function ###
        # range_ll = -0.5 * np.power(self.range_measurement - dist, 2.0) / self.measurement_cov
        _, _, _, _, _, eta = self.fft.analyze(range_ll.reshape(self.fft.spatial_grid_size))
        measurement_belief = SE2.from_eta(eta, self.fft)
        return measurement_belief

    def _update_beacon_idx(self) -> None:
        """
        Update beacon index, and cycle back to 0 if need be.
        """
        self.beacon_idx += 1
        if self.beacon_idx >= self.beacons.shape[0]:
            self.beacon_idx = 0

    def neg_log_likelihood(self, pose) -> np.ndarray:
        """
        Evaluate measurement distribution of a multivariate gaussian, note this is only evaluate over x-y plane.
        :param pose: Pose at which evaluate log likelihhod of measurement model
        :return ll: log probability of distribution determined by fourier coefficients (moments) at given pose
        """
        dist = np.linalg.norm(self.beacons[self.beacon_idx, :] - pose[0:2])
        ll = multivariate_normal.logpdf(self.range_measurement, mean=dist, cov=self.measurement_cov)
        return -ll
