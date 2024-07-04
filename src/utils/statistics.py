import numpy as np


def compute_weighted_mean(prob: np.ndarray,
                          poses: np.ndarray,
                          x: np.ndarray,
                          y: np.ndarray,
                          theta: np.ndarray) -> float:
    """
    Compute weighted mean of a distribution
    :return mean of distribution
    """
    # Compute mean
    prod = poses * prob.flatten()[:, None]
    prod = prod.reshape((x.shape[0], x.shape[1], x.shape[2], 3))
    # Integrate x, y, theta
    int_x = np.trapz(prod, x=x[..., None], axis=0)
    int_xy = np.trapz(int_x, x=y[0, :, :].squeeze()[..., None], axis=0)
    int_xyz = np.trapz(int_xy, x=theta[0, 0, :].squeeze()[..., None], axis=0)
    return int_xyz

def compute_mode(prob: np.ndarray, poses: np.ndarray) -> float:
    """
    Compute mode of the distribution
    :return mode of distribution
    """
    return poses[prob.argmax()]
