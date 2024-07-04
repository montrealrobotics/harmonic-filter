from typing import Optional

import numpy as np

from src.distributions.se2_distributions import SE2, SE2Gaussian, SE2MultimodalGaussian
from src.simulators.simulator_base import Simulator
from src.spectral.base_fft import FFTBase
from src.groups.se2_group import SE2Group
from src.utils.logging import get_logger

log = get_logger(__name__)


class SE2SimpleSimulator(Simulator):
    def __init__(self, start: SE2Group = SE2Group(),
                 step: SE2Group = SE2Group.from_parameters(0.1, 0.05, np.pi / 4),
                 samples: Optional[np.ndarray] = None,
                 fft: FFTBase = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.position = start
        self.step = step
        self.samples = samples
        self.fft = fft
        # If motion noise or measurement noise are zero, default to 0.1 and 0.4 respectively
        self.motion_cov = np.diag(self.motion_noise ** 2) if self.motion_noise.sum() != 0.0 else np.eye(3) * 1e-4
        self.measurement_cov = np.diag(self.measurement_noise ** 2) if self.measurement_noise.sum() != 0.0 else np.eye(
            3) * 5e-3

    def motion(self) -> SE2:
        """
        Simulate a motion with step.
        :return: SE2 distribution of relative predicted motion.
        """
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
        # Jitter measurement with noise and make sure theta is between 0 and 2pi
        noisy_measurement = self.position.parameters() + np.random.randn(3) * self.measurement_noise
        noisy_measurement[2] = noisy_measurement[2] % (2 * np.pi)
        return SE2MultimodalGaussian([noisy_measurement, noisy_measurement * np.asarray([1, -1, 1])],
                                     [self.measurement_cov, self.measurement_cov],
                                     samples=self.samples,
                                     fft=self.fft)
