from typing import Type

import numpy as np


class SE2Group:
    """
    Class representing the SE(2) group of rigid body transformations in R^2
    """

    def __init__(self, R: np.ndarray = np.eye(2), t: np.ndarray = np.zeros(2)):
        """
        Construct an SE2 object from a Rotation matrix R and translation t.

        :param R: rotaiton matrix (2,2)
        :param t: translation vector (2)
        """
        self.R = R
        self.t = t

    @classmethod
    def from_parameters(cls, x: float = 0., y: float = 0., theta: float = 0.0):
        """
        Construct an SE2 object from x, y, theta parameters.

        :param x: x component of translation
        :param y: y component of translation
        :param theta: rotation angle in radians
        :return: and SE2 object of the given parameters

        """
        R = np.zeros((2, 2))
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        t = np.array([x, y])
        return cls(R, t)

    def parameters(self) -> np.ndarray:
        """
        Get the x, y, theta parameters of the current transformation.

        :return: x, y theta parameters as a 3 dimensional vector.
        """
        return np.hstack([self.t, np.arctan2(self.R[1, 0], self.R[0, 0])])

    def __matmul__(self, other) -> Type['SE2Group']:
        """
        Group compostion of SE(2)
        """
        # return SE2Group(self.R @ other.R, other.R @ self.t + other.t)
        return SE2Group(self.R @ other.R, self.R @ other.t + self.t)

    def inv(self) -> Type['SE2Group']:
        """
        Inversion of the current transformation.

        :return: SE2 objection of the inverted transformation.
        """
        return SE2Group(self.R.T, -self.R.T @ self.t)
