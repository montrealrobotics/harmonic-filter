from typing import Optional
from omegaconf import DictConfig

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

from lie_learn.spectral.SE2FFT import SE2_FFT

from src.distributions.se2_distributions import SE2, SE2Gaussian
from src.filters.bayes_filter import BayesFilter
from src.groups.se2_group import SE2Group
from src.sampler.se2_sampler import se2_grid_samples
from src.simulators.se2_simple_simulator import SE2SimpleSimulator
from src.utils.se2_plotting import plot_se2_mean_filters
from src.utils.statistics import compute_weighted_mean
from src.utils import se2_plot_configs as plt_cfg
from src.utils.logging import seed_everything, get_logger, log_experiment_info, extras

log = get_logger(__name__)


def main(cfg: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed)
    results_path = os.path.join(cfg.results_path, datetime.datetime.now().isoformat())
    figures_path = os.path.join(results_path, "figures")
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    # Store config
    extras(cfg, results_path)

    n_samples = cfg.filter.n_samples
    grid_size = cfg.filter.grid_size
    var_motion, var_measurement = cfg.filter.var_motion, cfg.filter.var_measurement
    poses, x, y, theta = se2_grid_samples(grid_size)

    fft = SE2_FFT(spatial_grid_size=grid_size,
                  interpolation_method='spline',
                  spline_order=2,
                  oversampling_factor=3)

    mu_1 = np.array([0.0, -0.15, 0])
    cov_1 = np.diag(cfg.filter.var_prior)
    # Motion and measurement noise
    motion_noise, measurement_noise = np.ones(3) * np.sqrt(var_motion), np.ones(3) * np.sqrt(var_measurement)
    prior = SE2Gaussian(mu_1, cov_1, samples=poses, fft=fft)
    filter = BayesFilter(distribution=SE2, prior=prior)
    simulator = SE2SimpleSimulator(
        start=SE2Group.from_parameters(*mu_1),
        step=SE2Group.from_parameters(0.025, 0.01, np.pi / 20.0),
        samples=poses,
        fft=fft,
        motion_noise=motion_noise,
        measurement_noise=measurement_noise
    )
    it = 1
    trajectories = dict(HEF=np.zeros((n_samples, 3)), GT=np.zeros((n_samples, 3)))
    # while True:
    for i in range(n_samples):
        # Predict step
        belief_hat = filter.prediction(motion_model=simulator.motion())
        # Update step
        posteriori_hat, measurement_hat = filter.update(measurement_model=simulator.measurement())
        harmonic_pos_pose = compute_weighted_mean(posteriori_hat.prob, poses, x, y, theta)
        trajectories['HEF'][i] = harmonic_pos_pose

        gt_pose = simulator.position.parameters()
        trajectories['GT'][i] = gt_pose
        legend = [rf"Belief", rf"Measurement", rf"Posteriori"]
        axes = plot_se2_mean_filters(
            [belief_hat.prob.real, measurement_hat.prob.real, posteriori_hat.prob.real],
            x, y, theta,
            samples=trajectories, iteration=i,
            level_contours=True, contour_titles=legend, config=plt_cfg.CONFIG_MEAN_SE2_F)
        for ax in axes:
            # ax.set_aspect(1 / ax.get_data_ratio())  # make axes square
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
        # Extend title and legend
        axes[3].set_title(f"{axes[3].title.get_text()} - Step: {it}")
        plt.savefig(f"{figures_path}/se2_bayes_filter{it:03d}.png")
        # plt.show()
        it += 1
    # TODO - need a video compiler for this guy too...
    # Log information to the logger (if available)
    log_experiment_info(cfg, results_path)

    return 0
