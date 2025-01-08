# Code adapted from https://github.com/contagon/iekf/tree/master
from typing import Tuple

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
from src.groups.se2_group import SE2Group

class RangeIEKF:

    def __init__(self, system, prior, prior_cov):
        """The newfangled Invariant Extended Kalman Filter

        Args:
            system    (class) : The system to run the iEKF on. It will pull Q, R, f, h, F, H from this.
            mu0     (ndarray) : Initial starting point of system
            sigma0  (ndarray) : Initial covariance of system"""
        self.sys = system
        self.pose: SE2Group = SE2Group.from_parameters(*prior)
        self.state_cov: np.ndarray = prior_cov
    
    def prediction(self, step: np.ndarray, step_cov: np.ndarray, dt: float):
        """
        Runs prediction step of iEKF.

        Args:
            u       (k ndarray) : control taken at this step

        Returns:
            mu    (nxn ndarray) : Propagated state
            sigma (nxn ndarray) : Propagated covariances
        """
        # Extract d and theta from step
        d = np.sqrt(step[0]**2 + step[1]**2)
        theta = step[2]
        u = np.array([d, 0, theta])
        # Compute prediction
        self.pose = self.sys.f_lie(self.pose, u)
        # Update covariance matrix
        adj_u = self.sys.adjoint(inv(expm(self.sys.carat(np.array([u[0], 0, u[1]])))) )
        self.state_cov = adj_u @ self.state_cov @ adj_u.T + step_cov * dt

        return self.pose, self.state_cov
