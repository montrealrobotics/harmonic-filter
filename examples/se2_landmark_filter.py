from typing import Optional
from omegaconf import DictConfig, OmegaConf

import json
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from copy import deepcopy

from lie_learn.spectral.SE2FFT import SE2_FFT

from src.distributions.se2_distributions import SE2, SE2Gaussian
from src.filters.bayes_filter import BayesFilter
from src.filters.range_ekf import RangeEKF
from src.filters.range_hf import RangeHF
from src.filters.range_pf import RangePF
from src.groups.se2_group import SE2Group
from src.sampler.se2_sampler import se2_grid_samples
from src.simulators.se2_range_simulator import SE2RangeSimulator
from src.utils.create_video import create_mp4
from src.utils.numpy_json import NumpyEncoder
from src.utils.se2_plotting import (
    plot_se2_mean_filters,
    plot_se2_filters,
    plot_error_xy_trajectory,
    plot_neg_log_likelihood,
)
from src.utils.statistics import compute_weighted_mean, compute_mode
from src.utils import se2_plot_configs as plt_cfg
from src.utils.logging import seed_everything, get_logger, log_experiment_info, extras

log = get_logger(__name__)


def main(cfg: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed)
    results_path = os.path.join(cfg.results_path, datetime.datetime.now().isoformat())
    figures_path = os.path.join(results_path, "figures")
    others_path = os.path.join(results_path, "others")
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
        os.makedirs(others_path)

    # Store config
    extras(cfg, others_path)

    n_samples = cfg.filter.n_samples
    grid_size = cfg.filter.grid_size
    var_motion, var_measurement = cfg.filter.var_motion, cfg.filter.var_measurement
    poses, x, y, theta = se2_grid_samples(grid_size)

    fft = SE2_FFT(
        spatial_grid_size=grid_size,
        interpolation_method="spline",
        spline_order=2,
        oversampling_factor=3,
    )

    mu_1 = np.array([0.0, -0.15, 0])
    cov_1 = np.diag(cfg.filter.var_prior)
    # Motion and measurement noise
    motion_noise, measurement_noise = np.ones(3) * np.sqrt(var_motion), np.sqrt(
        var_measurement
    )
    prior = SE2Gaussian(mu_1, cov_1, samples=poses, fft=fft)
    prior.normalize()
    filter = BayesFilter(distribution=SE2, prior=prior)
    # Define Kalman Filter as baseline
    ekf = RangeEKF(prior=mu_1, prior_cov=cov_1)
    hf = RangeHF(prior=mu_1, prior_cov=cov_1, grid_samples=poses, grid_size=grid_size)
    pf = RangePF(prior=mu_1, prior_cov=cov_1, n_particles=np.prod(grid_size))
    simulator = SE2RangeSimulator(
        start=SE2Group.from_parameters(*mu_1),
        step=SE2Group.from_parameters(0.01, 0.00, np.pi / 40.0),
        samples=poses,
        fft=fft,
        motion_noise=motion_noise,
        measurement_noise=measurement_noise,
    )
    # Container to store filters' estimate
    trajectories = dict(
        HEF=np.zeros((n_samples, 3)),
        EKF=np.zeros((n_samples, 3)),
        HistF=np.zeros((n_samples, 3)),
        PF=np.zeros((n_samples, 3)),
        Measurement=np.zeros((n_samples, 3)),
        GT=np.zeros((n_samples, 3)),
    )
    trajectories_mode = deepcopy(trajectories)
    # Populate first position with prior pose
    for key in trajectories.keys():
        trajectories[key][0] = simulator.position.parameters()
        trajectories_mode[key][0] = simulator.position.parameters()
    nll = dict(HEF=[], EKF=[], HistF=[], PF=[], Measurement=[])
    # Populate nll with prior estimate wrt to GT
    gt_pose = simulator.position.parameters()
    # Assume this prior for measurement 
    nll["Measurement"].append(-np.log(0.5))
    nll["HEF"].append(
        filter.neg_log_likelihood(prior.eta, prior.l_n_z, gt_pose, fft).item()
    )
    nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
    nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
    nll["HistF"].append(hf.neg_log_likelihood(gt_pose))

    for it in tqdm(
        range(1, n_samples), total=n_samples - 1, desc="Filtering range simulator..."
    ):
        # Predict step
        motion_distribution = simulator.motion()
        belief_hat = filter.prediction(motion_model=motion_distribution)
        ekf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        hf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        pf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        # Update step
        measurement_distribution = simulator.measurement()
        # Obtain ground truth
        gt_pose = simulator.position.parameters()
        trajectories["GT"][it] = trajectories_mode["GT"][it] = gt_pose
        nll["Measurement"].append(simulator.neg_log_likelihood(gt_pose))
        # Compute mean of observation model
        trajectories["Measurement"][it] = compute_weighted_mean(
            measurement_distribution.prob, poses, x, y, theta
        )
        trajectories_mode["Measurement"][it] = compute_mode(measurement_distribution.prob, poses)
        # Harmonic filter
        posteriori_hat, measurement_hat = filter.update(measurement_model=measurement_distribution)
        # Compute weighted mean and ll of ground truth
        harmonic_pos_pose = compute_weighted_mean(
            posteriori_hat.prob, poses, x, y, theta
        )
        trajectories["HEF"][it] = harmonic_pos_pose
        harmonic_mode_pose = compute_mode(posteriori_hat.prob, poses)
        trajectories_mode["HEF"][it] = harmonic_mode_pose
        nll['HEF'].append(filter.neg_log_likelihood2(posteriori_hat.energy, posteriori_hat.l_n_z, gt_pose, grid_size).item())
        # EKF filter
        ekf_pos_pose, ekf_pos_cov = ekf.update(
            landmarks=simulator.beacons[simulator.beacon_idx, :2].reshape(1, 2),
            observations=np.asarray([simulator.range_measurement]),
            observations_cov=[1e-3],
        )
        trajectories["EKF"][it] = trajectories_mode["EKF"][it] = ekf_pos_pose
        nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
        # HF filter
        hf_pos_pose = hf.update(
            landmarks=simulator.beacons[simulator.beacon_idx, :2].reshape(1, 2),
            observations=np.asarray([simulator.range_measurement]),
            observations_cov=[1e-3],
        )
        trajectories["HistF"][it] = hf_pos_pose
        hf_mode_pose = hf.compute_mode()
        trajectories_mode["HistF"][it] = hf_mode_pose
        nll["HistF"].append(hf.neg_log_likelihood(gt_pose))
        # PF filter
        pf_pos_pose = pf.update(
            landmarks=simulator.beacons[simulator.beacon_idx, :2].reshape(1, 2),
            observations=np.asarray([simulator.range_measurement]),
            observations_cov=[1e-3],
        )
        trajectories["PF"][it] = pf_pos_pose
        pf_mode_pose = pf.compute_mode()
        trajectories_mode["PF"][it] = pf_mode_pose
        nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
        log.info(f"lnz: {posteriori_hat.l_n_z} at iteration {it}")
        # Plotting
        legend = [rf"Predicted belief", rf"Measurement Likelihood", rf"Posterior belief"]
        axes_means = plot_se2_mean_filters(
            [belief_hat.prob.real, measurement_hat.prob.real, posteriori_hat.prob.real],
            x,
            y,
            theta,
            samples=trajectories,
            iteration=it,
            beacons=simulator.beacons[:, :2],
            level_contours=False,
            contour_titles=legend,
            config=plt_cfg.CONFIG_MEAN_SE2_LF,
        )
        axes_modes = plot_se2_mean_filters(
            [belief_hat.prob.real, measurement_hat.prob.real, posteriori_hat.prob.real],
            x,
            y,
            theta,
            samples=trajectories_mode,
            iteration=it,
            beacons=simulator.beacons[:, :2],
            level_contours=False,
            contour_titles=legend,
            config=plt_cfg.CONFIG_MEAN_SE2_LF,
        )
        axes_filters = plot_se2_filters(
            {
                "HEF": [harmonic_pos_pose, posteriori_hat.prob.real, harmonic_mode_pose],
                "EKF": [ekf_pos_pose, ekf_pos_cov, ekf_pos_pose],
                "PF": [pf_pos_pose, pf.particles, pf_mode_pose],
                "HistF": [hf_pos_pose, hf.prior.reshape(grid_size), hf_mode_pose],
                "GT": [gt_pose, None],
            },
            x,
            y,
            theta,
            simulator.beacons[:, :2],
            titles=[
                f"Harmonic Exponential Filter",
                f"Extended Kalman Filter",
                f"Particle Filter",
                f"Histogram Filter",
            ],
            config=plt_cfg.CONFIG_FILTERS_SE2_LF,
        )
        for ax_mean, ax_mode, ax_filter in zip(axes_means, axes_modes, axes_filters):
            # ax.set_aspect(1 / ax.get_data_ratio())  # make axes square
            ax_mean.set_xlim(-0.5, 0.5)
            ax_mean.set_ylim(-0.5, 0.5)
            ax_mode.set_xlim(-0.5, 0.5)
            ax_mode.set_ylim(-0.5, 0.5)
            ax_filter.set_xlim(-0.5, 0.5)
            ax_filter.set_ylim(-0.5, 0.5)
            ax_filter.scatter(
                simulator.beacons[simulator.beacon_idx, 0],
                simulator.beacons[simulator.beacon_idx, 1],
                c="y",
                marker="o",
                s=80,
                alpha=0.8,
                zorder=2,
            )
        # Add which beacon is active, this is extra information needed only to this example
        axes_means[3].scatter(
            simulator.beacons[simulator.beacon_idx, 0],
            simulator.beacons[simulator.beacon_idx, 1],
            c="y",
            marker="o",
            s=80,
            alpha=0.8,
            zorder=2,
        )
        axes_modes[3].scatter(
            simulator.beacons[simulator.beacon_idx, 0],
            simulator.beacons[simulator.beacon_idx, 1],
            c="y",
            marker="o",
            s=80,
            alpha=0.8,
            zorder=2,
        )
        axes_means[3].set_title(f"Mean estimate - step: {it}", fontdict={'fontsize': 18})
        axes_modes[3].set_title(f"MAP estimate - step: {it}", fontdict={'fontsize': 18})
        # plt.show()
        # Save means' figure
        plt.figure(1)
        plt.savefig(f"{figures_path}/se2_main{it:03d}.png")
        plt.close()
        plt.figure(2)
        plt.savefig(f"{figures_path}/se2_map{it:03d}.png")
        plt.close()
        # Save filters' figure
        plt.figure(3)
        plt.savefig(f"{figures_path}/se2_filters{it:03d}.png")
        plt.close()

    # Plot log-likelihood of each estimator and the ground truth
    plot_neg_log_likelihood(nll, config=plt_cfg.CONFIG_LL_SE2_LF)
    plt.savefig(f"{others_path}/se2_nll.png")
    # plt.show()
    plt.close()
    # Plot trajectory
    ax, metrics = plot_error_xy_trajectory(
        trajectories,
        1.0,
        0.0,
        0.0,
        landmarks=simulator.beacons,
        config=plt_cfg.CONFIG_TRAJ_SE2_LF,
        x_y_limits=[-0.5, 0.5, -0.5, 0.5],
    )
    _, metrics_mode = plot_error_xy_trajectory(
        trajectories_mode,
        1.0,
        0.0,
        0.0,
        landmarks=simulator.beacons,
        config=plt_cfg.CONFIG_TRAJ_SE2_LF,
        x_y_limits=[-0.5, 0.5, -0.5, 0.5],
    )

    ax.set_title(f"Trajectory estimates - {n_samples} steps", fontdict={'fontsize': 18})
    # Print metrics
    table = PrettyTable()
    table.float_format = "6.3"
    table.title = 'Results w/ mean'
    table.field_names = ["Filter", "RMSE", "Mean", "Std", "MeanNLL"]
    table_mode = deepcopy(table)
    table_mode.title = 'Results w/ mode'
    metrics_dict = {}
    for k, v in metrics.items():
        map_filter = metrics_mode[k]
        table.add_row([k, v[0], v[1], v[2], np.mean(nll[k])])
        table_mode.add_row([k, map_filter[0], map_filter[1], map_filter[2], np.mean(nll[k])])
        metrics_dict[k] = {
            table.field_names[1]: v[0],
            table.field_names[2]: v[1],
            table.field_names[3]: v[2],
            "RMSE_MAP": map_filter[0],
            "Mean_MAP": map_filter[1],
            "Std_MAP": map_filter[2],
            table.field_names[4]: np.mean(nll[k]),
        }

    with open(f"{others_path}/results.json", "w") as f:
        json.dump(
            {
                "trajectories": trajectories,
                "trajectoris_mode": trajectories_mode,
                "metrics": metrics_dict,
                "neg_log_likelihood": nll,
                "params": OmegaConf.to_container(cfg),
            },
            f,
            cls=NumpyEncoder,
            indent=4,
        )

    print(table)
    print(table_mode)
    plt.savefig(f"{others_path}/se2_traj.png")
    # plt.show()
    plt.close()
    # Create video
    if cfg.get("duration"):
        create_mp4(results_path, "result.mp4", duration=cfg.duration)
    # Log information to the logger (if available)
    log_experiment_info(cfg, results_path)

    return 0
