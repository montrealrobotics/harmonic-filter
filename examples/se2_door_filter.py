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

from src.simulators.se2_door_dataset import SE2DoorDataset
from src.distributions.se2_distributions import SE2, SE2Gaussian
from src.filters.bayes_filter import BayesFilter
from src.filters.range_ekf import BearingEKF
from src.filters.range_pf import BearingPF
from src.filters.range_hf import BearingHF
from src.sampler.se2_sampler import se2_grid_samples
from src.utils.create_video import create_mp4
from src.utils.numpy_json import NumpyEncoder
from src.utils.se2_plotting import (
    plot_se2_mean_filters,
    plot_error_xy_trajectory,
    plot_se2_filters,
    plot_neg_log_likelihood,
    plot_angles,
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

    # Hyperparameters
    grid_size = cfg.filter.grid_size
    poses, x, y, theta = se2_grid_samples(grid_size)

    fft = SE2_FFT(
        spatial_grid_size=grid_size,
        interpolation_method="spline",
        spline_order=2,
        oversampling_factor=3,
    )

    scaling_factor = cfg.filter.scaling_factor
    d_door2pose = cfg.filter.d_door2pose
    doors_blacklist = cfg.filter.doors_blacklist
    offset_x, offset_y = cfg.filter.offset_x, cfg.filter.offset_y
    # Define motion noise and measurement noise
    var_motion = cfg.filter.var_motion
    motion_noise = np.ones(3) * np.sqrt(var_motion)
    measurement_noise = np.sqrt(cfg.filter.var_measurement)
    simulator = SE2DoorDataset(
        data_path=cfg.data_dir,
        fft=fft,
        d_door2pose=d_door2pose,
        scaling_factor=scaling_factor,
        offset_x=offset_x,
        offset_y=offset_y,
        doors_blacklist=doors_blacklist,
        motion_noise=motion_noise,
        samples=poses,
        measurement_noise=measurement_noise,
    )

    # Define prior and create filter
    mu_prior = simulator.position.parameters()
    cov_prior = np.diag(
        cfg.filter.var_prior
    )  # We are not certain about the orientation
    prior = SE2Gaussian(mu_prior, cov_prior, samples=poses, fft=fft)
    prior.normalize()
    filter = BayesFilter(distribution=SE2, prior=prior)
    ekf = BearingEKF(prior=mu_prior, prior_cov=cov_prior)
    pf = BearingPF(
        prior=mu_prior,
        prior_cov=cov_prior,
        n_particles=np.prod(grid_size),
        d_door2pose=d_door2pose,
    )
    hf = BearingHF(
        prior=mu_prior,
        prior_cov=cov_prior,
        grid_samples=poses,
        grid_size=grid_size,
        d_door2pose=d_door2pose,
    )

    trajectories = dict(
        HEF=np.zeros((simulator.n_samples, 3)),
        EKF=np.zeros((simulator.n_samples, 3)),
        HistF=np.zeros((simulator.n_samples, 3)),
        PF=np.zeros((simulator.n_samples, 3)),
        Measurement=np.zeros((simulator.n_samples, 3)),
        GT=np.zeros((simulator.n_samples, 3)),
    )
    trajectories_mode = deepcopy(trajectories)
    # Populate first position with prior pose
    for key in trajectories.keys():
        trajectories[key][0] = simulator.position.parameters()
        trajectories_mode[key][0] = simulator.position.parameters()
    nll = dict(HEF=[], EKF=[], HistF=[], PF=[], Measurement=[])
    # Populate nll with prior estimate wrt to GT
    gt_pose = simulator.position.parameters()
    nll["Measurement"].append(-np.log(0.5))
    nll['HEF'].append(filter.neg_log_likelihood2(prior.energy, prior.l_n_z, gt_pose, grid_size).item())
    nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
    nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
    nll["HistF"].append(hf.neg_log_likelihood(gt_pose))

    for it in tqdm(
        range(275),
        total=275,
        desc="Filtering door dataset...",
    ):
        ### Predict step ###
        motion_distribution = simulator.motion()
        belief_hat = filter.prediction(motion_model=motion_distribution)
        ekf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        pf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        hf.prediction(
            step=motion_distribution.mu,
            step_cov=np.linalg.inv(motion_distribution.inv_cov),
        )
        ### Update step ###
        measurement_distribution = simulator.measurement()
        gt_pose = simulator.gt_bins[simulator.iteration]
        trajectories["GT"][it] = trajectories_mode["GT"][it] = gt_pose
        trajectories["Measurement"][it] = compute_weighted_mean(
            measurement_distribution.prob, poses, x, y, theta
        )
        trajectories_mode["Measurement"][it] = compute_mode(measurement_distribution.prob, poses)
        # Compute log-likelihood of GT given the measurement
        nll["Measurement"].append(simulator.neg_log_likelihood(gt_pose))
        # Harmonic filter
        posteriori_hat, measurement_hat = filter.update(
            measurement_model=measurement_distribution
        )
        harmonic_pos_pose = compute_weighted_mean(
            posteriori_hat.prob, poses, x, y, theta
        )
        trajectories["HEF"][it] = harmonic_pos_pose
        harmonic_mode_pose = compute_mode(posteriori_hat.prob, poses)
        trajectories_mode["HEF"][it] = harmonic_mode_pose
        nll['HEF'].append(filter.neg_log_likelihood2(posteriori_hat.energy, posteriori_hat.l_n_z, gt_pose, grid_size).item())
        # EKF filter
        ekf_pos_pose, ekf_pos_cov = ekf.update(
            landmarks=np.array(simulator.doors),
            observations=np.array(simulator.bearing_bins[simulator.iteration]),
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        trajectories["EKF"][it] = trajectories_mode["EKF"][it] = ekf_pos_pose
        nll["EKF"].append(ekf.neg_log_likelihood(gt_pose))
        # PF filter
        pf_pos_pose = pf.update(
            landmarks=np.array(simulator.doors),
            map_mask=simulator.map_mask_unprocessed,
            observations=simulator.bearing_bins[simulator.iteration],
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        trajectories["PF"][it] = pf_pos_pose
        pf_mode_pose = pf.compute_mode()
        trajectories_mode["PF"][it] = pf_mode_pose
        nll["PF"].append(pf.neg_log_likelihood(gt_pose, (-0.5, 0.5), grid_size))
        # HF filter
        hf_pos_pose = hf.update(
            landmarks=np.array(simulator.doors),
            map_mask=simulator.map_mask,
            observations=simulator.bearing_bins[simulator.iteration],
            observations_cov=np.ones(len(simulator.doors)) * simulator.measurement_cov,
        )
        trajectories["HistF"][it] = hf_pos_pose
        hf_mode_pose = hf.compute_mode()
        trajectories_mode["HistF"][it] = hf_mode_pose
        nll["HistF"].append(hf.neg_log_likelihood(gt_pose))

        # Plotting the bearing - only for debuggin!
        # plot_angles(gt_pose, bearings=simulator.bearing_bins[simulator.iteration],
                    # landmarks=simulator.doors, map_array=simulator.map_array)
        # plt.savefig(f"{figures_path}/se2_mean{it:03d}.png")
        # plt.show()
        legend = [rf"Predicted belief", rf"Measurement Likelihood", rf"Posterior belief"]
        axes_means = plot_se2_mean_filters(
            [belief_hat.prob.real, measurement_hat.prob.real, posteriori_hat.prob.real],
            x,
            y,
            theta,
            samples=trajectories,
            iteration=it,
            beacons=np.array(simulator.doors),
            level_contours=False,
            contour_titles=legend,
            config=plt_cfg.CONFIG_MEAN_SE2_UWB,
        )
        axes_modes = plot_se2_mean_filters(
            [belief_hat.prob.real, measurement_hat.prob.real, posteriori_hat.prob.real],
            x,
            y,
            theta,
            samples=trajectories_mode,
            iteration=it,
            beacons=np.array(simulator.doors),
            level_contours=False,
            contour_titles=legend,
            config=plt_cfg.CONFIG_MEAN_SE2_UWB,
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
            np.array(simulator.doors),
            titles=[
                f"Harmonic Exponential Filter",
                f"Extended Kalman Filter",
                f"Particle Filter",
                f"Histogram Filter",
            ],
            config=plt_cfg.CONFIG_FILTERS_SE2_UWB,
        )
        for ax_mean, ax_mode, ax_filter in zip(axes_means, axes_modes, axes_filters):
            # Plot landmark name on each beacon for filters' plot - this is specific of this example only
            for i in range(len(simulator.doors)):
                ax_filter.text(
                    simulator.doors[i][0] + 5e-2,
                    simulator.doors[i][1],
                    f"D{i + 1}",
                    fontsize=12,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            # Plot map on both plots
            ax_filter.imshow(
                simulator.map_array[2],
                extent=[
                    simulator.map_array[0].min(),
                    simulator.map_array[0].max(),
                    simulator.map_array[1].min(),
                    simulator.map_array[1].max(),
                ],
                origin="upper",
                cmap=plt.cm.Greys_r,
                alpha=0.2,
                zorder=1,
            )
            ax_mean.set_xlim(-0.5, 0.5)
            ax_mean.set_ylim(-0.5, 0.5)
            ax_mode.set_xlim(-0.5, 0.5)
            ax_mode.set_ylim(-0.5, 0.5)
            ax_filter.set_xlim(-0.5, 0.5)
            ax_filter.set_ylim(-0.5, 0.5)
        # Plot landmark name on each beacon for main means plot - this is specific of this example only
        for i, b in enumerate(simulator.doors):
            axes_means[3].text(
                simulator.doors[i][0] + 2.5e-2,
                simulator.doors[i][1],
                f"D{i + 1}",
                fontsize=12,
                color="black",
            )
            axes_modes[3].text(
                simulator.doors[i][0] + 2.5e-2,
                simulator.doors[i][1],
                f"D{i + 1}",
                fontsize=12,
                color="black",
            )
        dead_reckoning = simulator.position.parameters()
        axes_means[3].scatter(dead_reckoning[0], dead_reckoning[1], marker='o', s=30,  c='k', zorder=4)
        axes_means[3].imshow(
            simulator.map_array[2],
            extent=[
                simulator.map_array[0].min(),
                simulator.map_array[0].max(),
                simulator.map_array[1].min(),
                simulator.map_array[1].max(),
            ],
            origin="upper",
            cmap=plt.cm.Greys_r,
            alpha=0.8,
            zorder=2,
        )
        axes_modes[3].scatter(dead_reckoning[0], dead_reckoning[1], marker='o', s=30,  c='k', zorder=4)
        axes_modes[3].imshow(
            simulator.map_array[2],
            extent=[
                simulator.map_array[0].min(),
                simulator.map_array[0].max(),
                simulator.map_array[1].min(),
                simulator.map_array[1].max(),
            ],
            origin="upper",
            cmap=plt.cm.Greys_r,
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
        # plt.show()

    # Plot log-likelihood of each estimator and the ground truth
    plot_neg_log_likelihood(nll, config=plt_cfg.CONFIG_LL_SE2_LF)
    plt.savefig(f"{others_path}/se2_nll.png")
    plt.close()
    # Plot trajectory
    ax, metrics = plot_error_xy_trajectory(
        trajectories,
        scaling_factor,
        offset_x,
        offset_y,
        landmarks=np.array(simulator.doors),
        config=plt_cfg.CONFIG_TRAJ_SE2_UWB,
        x_y_limits=[-18.5, 8.5, -8.0, 16.5],
    )
    _, metrics_mode = plot_error_xy_trajectory(
        trajectories_mode,
        scaling_factor,
        offset_x,
        offset_y,
        landmarks=np.array(simulator.doors),
        config=plt_cfg.CONFIG_TRAJ_SE2_UWB,
        x_y_limits=[-18.5, 8.5, -8.0, 16.5],
    )
    # Scale map coordinates and add map to plot
    map_x = (simulator.map_array[0] / scaling_factor) - offset_x
    map_y = (simulator.map_array[1] / scaling_factor) - offset_y
    ax.imshow(
        simulator.map_array[2],
        extent=[map_x.min(), map_x.max(), map_y.min(), map_y.max()],
        origin="upper",
        cmap=plt.cm.Greys_r,
        alpha=0.8,
        zorder=0,
    )
    ax.set_title(f"Trajectory estimates - {simulator.n_samples} steps", fontdict={'fontsize': 18})
    # Print metrics
    table = PrettyTable()
    table.float_format = "6.3"
    table.title = 'Results w/ mean'
    table.field_names = ["Filter", "RMSE", "Mean", "Std", "MeanNLL"]
    table_mode = deepcopy(table)
    table_mode.title = 'Results w/ mode'
    metrics_dict = {}
    for k, v in metrics.items():
        # Dead reckoning does not have ll
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
    plt.close()
    # Create video
    if cfg.get("duration"):
        create_mp4(results_path, "result.mp4", duration=cfg.duration)
    # Log information to the logger (if available)
    log_experiment_info(cfg, results_path)

    return 0.0
