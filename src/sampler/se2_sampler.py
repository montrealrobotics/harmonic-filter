from typing import Tuple

import numpy as np


def se2_grid_samples(size: Tuple[int] = (5, 5, 5),
                     lower_bound: float = -0.5,
                     upper_bound: float = 0.5) -> np.ndarray:
    xs = np.linspace(lower_bound, upper_bound, size[0], endpoint=False)
    ys = np.linspace(lower_bound, upper_bound, size[1], endpoint=False)
    ts = np.linspace(0., 2. * np.pi, size[2], endpoint=False)
    X, Y, T = np.meshgrid(xs, ys, ts, indexing='ij')
    poses = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    return poses, X, Y, T
