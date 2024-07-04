from typing import Optional, List

import os
from einops import rearrange
import numpy as np
from scipy.special import logsumexp
from copy import deepcopy

from scipy.stats import norm
from scipy.spatial.transform import Rotation as R

from src.distributions.se2_distributions import SE2, SE2Gaussian
from src.simulators.simulator_base import Simulator
from src.spectral.base_fft import FFTBase
from src.groups.se2_group import SE2Group
from src.utils.door_dataset_utils import (
    load_transforms,
    load_detections,
    load_map,
    load_map_array,
    preprocess_mask,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class SE2DoorDataset(Simulator):
    def __init__(
        self,
        data_path: str,
        d_door2pose: float = 0.1,
        scaling_factor: float = 1.0 / 25.0,
        offset_x: float = -2.19,
        offset_y: float = -2.19,
        doors_blacklist: List = [],
        samples: Optional[np.ndarray] = None,
        fft: FFTBase = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position: Optional[SE2Group] = None
        self.data_path = data_path
        # Used to scale motion step, UWB measurements and standard deviations
        self.d_door2pose = d_door2pose
        self.scaling_factor = scaling_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.doors_blacklist = doors_blacklist
        self.iteration = -1
        self.samples = samples
        self.fft = fft
        # Set motion and measurement covariance
        self.motion_cov = np.diag(self.motion_noise**2)
        self.measurement_cov = self.measurement_noise**2

        # Data containers
        self.initial_pose: Optional[np.ndarray] = None
        self.odom_bins: Optional[np.ndarray] = None
        self.gt_bins: Optional[np.ndarray] = None
        self.bearing_bins: Optional[np.ndarray] = None
        self.map_array: Optional[np.ndarray] = None
        self.map_mask: Optional[np.ndarray] = None
        # Used in particle filter
        self.map_mask_unprocessed: Optional[np.ndarray] = None
        self.doors: Optional[np.ndarray] = None
        # Setup dataset
        self.setup_data()
        # Define number of samples and start position of the agent
        self.n_samples = len(self.odom_bins)
        self.position = SE2Group.from_parameters(*self.initial_pose)
        log.info(f"Number of samples: {self.n_samples}")
        log.info(f"Start position: {self.position.parameters()}")

    def motion(self) -> SE2:
        """
        Simulate a motion with step.
        :return: SE2 distribution of relative predicted motion.
        """
        # Update iterations
        self.iteration += 1
        step = SE2Group.from_parameters(
            self.odom_bins[self.iteration][0] * 1.2,
            self.odom_bins[self.iteration][1] * 1.2,
            self.odom_bins[self.iteration][2],
        )
        # print(f"Current step: {step.parameters()}")
        self.position = self.position @ step

        return SE2Gaussian(
            step.parameters(), self.motion_cov, samples=self.samples, fft=self.fft
        )

    def measurement(self) -> SE2:
        """
        Simulate measurement
        :return: current position as the measurement as a vector of [x, y, theta].
        """
        # Observation z_t
        bearing_measurements = np.array(self.bearing_bins[self.iteration])
        observations_std = np.ones(len(self.doors)) * self.measurement_noise
        energy = np.log(1e-9)
        ### independent measurements but p(z_{t,i} | x_t, m) is a mixture of n_doors components ###
        for i, obs in enumerate(bearing_measurements):
            diff = rearrange(self.doors, "n m -> n 1 m") - rearrange(
                self.samples[:, :2], "p m -> 1 p m"
            )
            angle = np.arctan2(diff[:, :, 1], diff[:, :, 0]) - (
                (self.samples[:, 2] + np.pi) % (2 * np.pi) - np.pi
            )
            # Wrap angle.
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            diff_angle = obs - angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            mixture = self.map_mask * norm(diff_angle, rearrange(observations_std, "n -> n 1")).pdf(0.0)
            mixture = mixture.max(0) + 1e-8
            # Max along components dimension
            energy = np.maximum(energy, np.log(mixture))

        # Normalize
        #energy -= logsumexp(energy)
        _, _, _, _, _, eta = self.fft.analyze(
            energy.reshape(self.fft.spatial_grid_size)
        )
        measurement_belief = SE2.from_eta(eta, self.fft)
        return measurement_belief

    def neg_log_likelihood(self, pose) -> np.ndarray:
        """
        Evaluate measurement distribution of a multivariate gaussian, note this is only evaluate over x-y plane. Also,
        this should be called after calling the measurement() function
        :param pose: Pose at which evaluate log likelihhod of measurement model
        :return ll: log probability of distribution determined by fourier coefficients (moments) at given pose
        """
        bearing_measurements = np.array(self.bearing_bins[self.iteration])
        observations_std = np.ones(len(self.doors)) * self.measurement_noise
        log_prob = 0.0
        ### independent measurements but p(z_{t,i} | x_t, m) is a mixture of n_doors components ###
        for i, obs in enumerate(bearing_measurements):
            diff = np.array(self.doors) - pose[:2]
            angle = np.arctan2(diff[:, 1], diff[:, 0]) - pose[2]
            # Wrap angle.
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            diff_angle = obs - angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            # Get weight for current sample
            id = np.argmin(np.linalg.norm((self.samples[:, :2] - pose[:2]), axis=1))
            weight = self.map_mask[id]
            # Sum along components dimension
            mixture = weight * norm(diff_angle, observations_std).pdf(0.0)
            mixture = mixture.max() + 1e-8
            # Max along components dimension
            log_prob += np.log(mixture)
        return -log_prob

    def setup_data(self):
        # Read data
        odom_data = load_transforms(
            os.path.join(self.data_path, "transforms/odom_to_imu_link.json")
        )
        gt_data = load_transforms(
            os.path.join(self.data_path, "transforms/map_to_imu_link.json")
        )
        detections = load_detections(
            os.path.join(self.data_path, "images/detections.csv"),
            os.path.join(
                self.data_path, "calibration/extrinsics_cam0_imu0-camchain-imucam.yaml"
            ),
            self.doors_blacklist
        )
        # Load map
        self.map_array = load_map_array(os.path.join(self.data_path, "map/lab.pgm"))
        self.map_mask = load_map_array(os.path.join(self.data_path, "map/lab_mask.pgm"))
        self.map_mask_unprocessed = deepcopy(self.map_mask)
        self.doors = load_map()

        # Get important timestamps
        first_timestamp = odom_data[0]["timestamp"]
        odom_timestamps = np.array([odom["timestamp"] for odom in odom_data])
        gt_timestamps = np.array([gt["timestamp"] for gt in gt_data])

        # Get closest timestapm from odom to gt
        i_odom_gt = np.argmin(np.abs(gt_timestamps - first_timestamp))
        self.initial_pose = self._to_se2(gt_data[i_odom_gt]["transform"])

        self.odom_bins = []
        odom_last_idx = 0
        self.gt_bins = []
        self.bearing_bins = []
        for d in detections:
            t = d["timestamp"]
            self.bearing_bins.append(d["bearings"])
            # Find closest element from detection to gt and store it
            i_d_gt = np.argmin(np.abs(gt_timestamps - t))
            transform = gt_data[i_d_gt]["transform"]
            self.gt_bins.append(self._to_se2(transform))
            # Find closest element from detection to odom and store it
            i_d_odom = np.argmin(np.abs(odom_timestamps - t))
            self.odom_bins.append(
                self._compute_relative(
                    odom_data[odom_last_idx]["transform"],
                    odom_data[i_d_odom]["transform"],
                )
            )
            odom_last_idx = i_d_odom
        # Check all dimensions are correct
        assert (
            len(self.odom_bins) == len(self.gt_bins) == len(self.bearing_bins)
        ), "All bins must have the same number of rows. Consider changing delta_t"
        # Scale data
        self._scale_data()
        # Once everything is scaled, preprocess mask
        self.map_mask = preprocess_mask(self.map_mask, self.samples, )


    def _compute_relative(self, transform_AB, transform_AC):
        """
        Compute relative transform between to transforms, and flatten to se2.
        """
        R_BA = R.from_quat(transform_AB[3:7]).inv()
        t_BA = -R_BA.apply(transform_AB[0:3])

        R_AC = R.from_quat(transform_AC[3:7])
        t_AC = transform_AC[0:3]

        R_BC = R_BA * R_AC
        t_BC = t_BA + R_BA.apply(t_AC)

        return np.hstack([t_BC[0:2], R_BC.as_euler("ZYX")[0]])

    def _to_se2(self, se3_transform: np.ndarray) -> np.ndarray:
        """
        Flatten and se3 transform into an se2 transform in the xy plane.
        """
        r = R.from_quat(se3_transform[3:7])
        return np.hstack([se3_transform[0:2], r.as_euler("ZYX")[0]])

    def _scale_data(self):
        """
        Scale the data to fit in the map [-0.5, 0.5]
        """
        # Ground truth
        for i, gt in enumerate(self.gt_bins):
            new_gt = gt
            new_gt[0] = (new_gt[0] + self.offset_x) * self.scaling_factor
            new_gt[1] = (new_gt[1] + self.offset_y) * self.scaling_factor
            self.gt_bins[i] = new_gt
        # Odometry
        for i, odom in enumerate(self.odom_bins):
            new_odom = odom
            new_odom[0:2] *= self.scaling_factor
            self.odom_bins[i] = new_odom
        # Doors
        for i, d in enumerate(self.doors):
            new_door = d
            new_door[0] = (new_door[0] + self.offset_x) * self.scaling_factor
            new_door[1] = (new_door[1] + self.offset_y) * self.scaling_factor
            self.doors[i] = new_door
        # Map
        self.map_array[0] = (self.map_array[0] + self.offset_x) * self.scaling_factor
        self.map_array[1] = (self.map_array[1] + self.offset_y) * self.scaling_factor
        self.map_mask[0] = (self.map_mask[0] + self.offset_x) * self.scaling_factor
        self.map_mask[1] = (self.map_mask[1] + self.offset_y) * self.scaling_factor
        self.map_mask_unprocessed[0] = (self.map_mask_unprocessed[0] + self.offset_x) * self.scaling_factor
        self.map_mask_unprocessed[1] = (self.map_mask_unprocessed[1] + self.offset_y) * self.scaling_factor

        # Initial Pose
        self.initial_pose[0] = ( self.initial_pose[0] + self.offset_x) * self.scaling_factor
        self.initial_pose[1] = ( self.initial_pose[1] + self.offset_y) * self.scaling_factor
