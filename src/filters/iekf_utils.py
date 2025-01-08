from typing import Optional

import numpy as np
from scipy.linalg import expm
from src.groups.se2_group import SE2Group

class IEKFUtils:
    # Code adapted from https://github.com/contagon/iekf/tree/master
    def __init__(self, Q = np.eye(3), R = np.eye(3), deltaT = 1.0):
        """The basic unicycle model. 

        Args:
            Q (3,3 nparray): Covariance of noise on state
            R (2x2 nparray): Covariance of noise on measurements"""
        self.Q = Q
        self.R = R
        self.deltaT = deltaT
        self.b = np.array([0, 0, 1])


    def f_lie(self, pose, step, noise=False):
        """Propagates state forward in Lie Group. Used for gen_data and IEKF.

        Args:
            state (3,3 ndarray) : X_n of model in Lie Group
            u     (3,3 ndarray) : U_n of model as a vector
            noise        (bool) : Whether or not to add noise. Defaults to False.

        Returns:
            X_{n+1} (3,3 ndarray)"""
        # do actual propagating
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Q)
        else:
            w = np.zeros(3)
        dx = expm(self.carat(np.array([step[0], 0, step[1]] + w)))
        dx = SE2Group(dx[:2, :2], dx[:2, 2])
        return pose @ dx

    def h(self, state, noise=False):
        """Calculates measurement given a state. Note that the result is
            the same if it's in standard or Lie Group form, so we simplify into
            one function.
            
        Args:
            state (3 ndarray or 3,3 ndarray) : Current state in either standard or Lie Group form
            noise                     (bool) : Whether or not to add noise. Defaults to False.

            
        Returns:
            Z_n (3 ndarray or 3,3 ndarray)"""
        # using standard coordinates
        if state.shape == (3,):
            z = np.array([state[0], state[1]])
        # using Lie Group
        elif state.shape == (3,3):
            z = state @ self.b

        #add noise if needed
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(2), cov=self.R)
            z[:2] += w

        return z

    @staticmethod
    def carat(xi):
        """Moves an vector to the Lie Algebra se(3).

        Args:
            xi (3 ndarray) : Parametrization of Lie algebra

        Returns:
            xi^ (3,3 ndarray) : Element in Lie Algebra se(2)"""
        return np.array([[0,   -xi[2], xi[0]],
                        [xi[2], 0,     xi[1]],
                        [0,     0,     0]])

    @staticmethod
    def adjoint(xi):
        """Takes adjoint of element in SE(3)

        Args:
            xi (3x3 ndarray) : Element in Lie Group

        Returns:
            Ad_xi (3,3 ndarray) : Adjoint in SE(3)"""
        # make the swap
        xi[0,2], xi[1,2] = xi[1,2], -xi[0,2]
        return xi

class ExpSE2:
    def __init__(self, pose_matrix: Optional[SE2Group] = None, tau: Optional[np.ndarray] = None):
        if tau is not None:
            self.tau = tau
        elif pose_matrix is not None:
            self.g = np.eye(3)
            self.g[:2, :2] = pose_matrix.R
            self.g[:2, -1] = pose_matrix.t
            self.tau = self.log_map()
        else:
            raise NotImplementedError("Provide a pose in SE(2) or exp coordinates")

    @staticmethod
    def skew_symmetric_so2(theta):
        return np.asarray([[0, -theta], [theta, 0]])

    def hat_se2(self, tau: Optional[np.ndarray] = None):
        tau = tau if tau is not None else self.tau
        tau_hat = np.zeros((3, 3))
        tau_hat[:2, :2] = self.skew_symmetric_so2(tau[-1])
        tau_hat[0, -1], tau_hat[1, -1] = tau[0], tau[1]
        return tau_hat

    def vee_se2(self, tau_hat):
        return np.asarray([tau_hat[0], tau_hat[1], tau_hat[2]])

    def exp_map(self):
        theta = self.tau[-1]
        # Compute Jacobian SE(2)
        jac = (np.sin(theta) / theta) * np.eye(2) + \
              ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        # Obtain rotation matrix
        g = np.eye(3)
        traslation = (jac @ self.tau[:2][:, np.newaxis]).squeeze()  # Eq. (6-7)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        g[:2, :2] = rotation
        g[:2, -1] = traslation

        return SE2Group(rotation, traslation)

    def log_map(self):
        # Compute logmap SO(2)
        theta = np.arctan2(self.g[1, 0], self.g[0, 0])
        # Edge case when theta is zero
        if theta != 0.0:
            jac = (np.sin(theta) / theta) * np.eye(2) + \
                  ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        else:
            jac = np.eye(2)
        # Compute translation component
        rho = (np.linalg.inv(jac) @ self.g[:2, -1]).squeeze()
        tau = np.asarray([rho[0], rho[1], theta])
        return tau

    def __str__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg

    def __repr__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg
