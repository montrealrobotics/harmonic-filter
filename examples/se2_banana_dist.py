from typing import List

import numpy as np
import matplotlib.pyplot as plt

from lie_learn.spectral.SE2FFT import SE2_FFT

from src.distributions.se2_distributions import SE2, SE2Gaussian, SE2Square
from src.filters.bayes_filter import BayesFilter
from src.filters.range_ekf import RangeEKF
from src.filters.range_pf import RangePF
from src.filters.range_hf import RangeHF
from src.sampler.se2_sampler import se2_grid_samples
from src.utils.se2_plotting import plot_se2_bananas
from src.utils.statistics import compute_weighted_mean, compute_mode
from src.utils import se2_plot_configs as plt_cfg


def sample_square_particles(
    x_limits: List[float],
    y_limits: List[float],
    theta_limits: List[float],
    mu: np.ndarray,
    cov: np.ndarray,
    n_particles: int,
):
    # Sample a large number of samples as many will be discarted
    particles = (np.linalg.cholesky(cov) @ np.random.randn(3, n_particles * 50)).T + mu
    # Discard particles that are out-of-bounds
    x_mask = np.logical_and(
        x_limits[0] < particles[:, 0], particles[:, 0] < x_limits[1]
    )
    y_mask = np.logical_and(
        y_limits[0] < particles[:, 1], particles[:, 1] < y_limits[1]
    )
    t_mask = np.logical_and(
        theta_limits[0] < particles[:, 2], particles[:, 2] < theta_limits[1]
    )
    total_mask = np.logical_and(np.logical_and(x_mask, y_mask), t_mask)
    # Get only first n_samples
    particles = particles[total_mask][:n_particles]
    return particles


def main():
    size = (50, 50, 50)
    poses, x, y, theta = se2_grid_samples(size, lower_bound=-0.5, upper_bound=0.5)

    fft = SE2_FFT(
        spatial_grid_size=size,
        interpolation_method="spline",
        spline_order=2,
        oversampling_factor=3,
    )

    # Prior belief
    mu_belief = np.array([-0.25, 0.0, 0.0])
    cov_belief = np.diag([0.0011, 0.018, 0.001])
    # Limits for square plot
    x_limits = [-0.25, -0.15]
    y_limits = [-0.2, 0.2]
    theta_limits = [-0.1, 0.1]
    belief = SE2Square(
        x_limits=x_limits,
        y_limits=y_limits,
        theta_limits=theta_limits,
        scale=10.0,
        samples=poses,
        fft=fft,
    )
    # Step - Gaussian
    mu = np.array([0.1, 0.0, 0.0])
    cov = np.diag([0.01, 0.01, 20.0]) * 1e-2
    step = SE2Gaussian(mu, cov, samples=poses, fft=fft)

    # Create filters
    hef = BayesFilter(distribution=SE2, prior=belief)
    # Define Kalman Filter as baseline - Gaussian assumption does not allow for square dist.
    ekf = RangeEKF(prior=mu_belief, prior_cov=cov_belief)
    pf = RangePF(prior=mu_belief, prior_cov=cov_belief, n_particles=np.prod(size))
    # Re-sample particles in a square-like shape
    pf.particles = sample_square_particles(
        x_limits, y_limits, theta_limits, mu_belief, cov_belief, np.prod(size)
    )
    # Update prior of histogram to be square-like
    hf = RangeHF(
        prior=mu_belief, prior_cov=cov_belief, grid_samples=poses, grid_size=size
    )
    hf.prior = (belief.energy / belief.energy.sum()).flatten()

    # Prediction step on each distribution
    for _ in range(4):
        belief_hat = hef.prediction(motion_model=step)
        ekf.prediction(step=step.mu, step_cov=step.cov)
        pf.prediction(step=step.mu, step_cov=step.cov)
        hf.prediction(step=step.mu, step_cov=step.cov)

    # Get statistics of each filter
    hef_pose = compute_weighted_mean(belief_hat.prob, poses, x, y, theta)
    hef_mode = compute_mode(belief_hat.prob, poses)
    ekf_pose = ekf.pose.parameters()
    pf_pose = np.mean(pf.particles, axis=0)
    # Approximate mode of particle filter being particle closer to GT (we do not have updated weights here)
    gt = np.array([0.15, 0, 0])
    index = np.linalg.norm(pf.particles - gt, axis=1)
    pf_mode = pf.particles[index.argmin()]
    hf_pose = hf._compute_mean()
    hf_mode = hf.compute_mode()

    axes_filters = plot_se2_bananas(
        {
            "prior": [None, belief.prob.real],
            "step": [None, step.prob.real],
            "HEF": [hef_pose, belief_hat.prob.real, hef_mode],
            "EKF": [ekf_pose, ekf.state_cov, ekf_pose],
            "PF": [pf_pose, pf.particles, pf_mode],
            "HistF": [hf_pose, hf.prior.reshape(size), hf_mode],
        },
        x,
        y,
        theta,
        titles=[
            f"Prior belief",
            f"Step belief",
            f"Harmonic Exponential Filter",
            f"Extended Kalman Filter",
            f"Particle Filter",
            f"Histogram Filter",
        ],
        cfg=plt_cfg.CONFIG_FILTERS_SE2_UWB,
    )

    plt.show()

if __name__ == "__main__":
    main()
