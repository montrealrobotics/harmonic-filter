from typing import List
import os
import json
import yaml
from PIL import Image
import pandas as pd

import cv2
import numpy as np

DOORS = [[1155, 1297], [1166, 1181], [1284, 1066], [1453, 1078], [1315, 883],
         [1268, 753], [1163, 666], [1075, 549], [649, 275], [634, 390],
         [408, 509], [280, 638], [228, 796], [162, 762], [140, 888],
         [498, 899], [474, 991], [620, 1017], [626, 1128], [817, 1210], [769, 1264]]

RESOLUTION = 0.02

ORIGIN = np.array([-21.52, 20])


def undistort_pixels(pixel, map_x, map_y):
    return np.array([map_x[pixel[1], pixel[0]], map_y[pixel[1], pixel[0]]])


def load_detections(detections_path: str, calibration_path: str, doors_blacklist: List[int]) -> list:
    detections = pd.read_csv(detections_path)
    d_timestamps = detections['image_name'].unique()
    d_timestamps.sort()

    with open(calibration_path, 'r') as f:
        calibrations = yaml.safe_load(f)

    T_imu_cam = np.linalg.inv(np.array(calibrations["cam0"]["T_cam_imu"]))

    cam_K = np.eye(3)
    cam_K[0, 0] = calibrations["cam0"]["intrinsics"][0]
    cam_K[1, 1] = calibrations["cam0"]["intrinsics"][1]
    cam_K[0, 2] = calibrations["cam0"]["intrinsics"][2]
    cam_K[1, 2] = calibrations["cam0"]["intrinsics"][3]

    cam_K_inv = np.linalg.inv(cam_K)

    dist_coefs = np.array(calibrations["cam0"]["distortion_coeffs"])
    resolution = np.array(calibrations["cam0"]["resolution"])

    map_x, map_y = cv2.initUndistortRectifyMap(cam_K, dist_coefs, np.eye(3),
                                               cam_K, resolution, cv2.CV_32FC1)

    detections_in_imu_frame = []
    remove_doors = {f"d{d}" for d in doors_blacklist}

    for timestamp in d_timestamps:
        bearings = []
        detection = detections.loc[detections['image_name'] == timestamp]
        # bbox_x, bbox_y, bbox_width, bbox_height
        bboxes = detection.iloc[:, 1:5].to_numpy()
        class_id = detection.iloc[:, 0].to_numpy()
        for box, id in zip(bboxes, class_id):
            # Remove blacklisted doors
            if id in remove_doors:
                continue
            mid_pixel = np.array([box[0] + (box[2] / 2.), box[1] + (box[3] / 2.)])
            pixel = np.ones(3)
            pixel[0:2] = undistort_pixels(np.rint(mid_pixel).astype(int), map_x, map_y)
            vector_cam = np.ones(4)
            vector_cam[0:3] = cam_K_inv @ pixel
            direction_imu = T_imu_cam @ vector_cam
            bearings.append(np.arctan2(direction_imu[1], direction_imu[0]))
        if len(bearings) == 0:
            continue
        timestamp = float(timestamp[:-4].replace("_", "."))

        detections_in_imu_frame.append({"timestamp": timestamp,
                                        "bearings": bearings})
    # Sort detections by timestamp
    detections_in_imu_frame = sorted(detections_in_imu_frame,
                                     key=lambda detections: detections["timestamp"])
    # Update list of doors and remove those that are blacklisted
    global DOORS
    DOORS = [d for i, d in enumerate(DOORS) if i + 1 not in doors_blacklist]
    return detections_in_imu_frame


def load_transforms(odom_path: str) -> list:
    with open(odom_path, 'r') as f:
        data = json.load(f)

    list_data = []

    for d in data:
        transform = np.array([
            d["translation"]["x"],
            d["translation"]["y"],
            d["translation"]["z"],
            d["rotation"]["x"],
            d["rotation"]["y"],
            d["rotation"]["z"],
            d["rotation"]["w"]
        ])
        list_data.append({"timestamp": d["timestamp"],
                          "transform": transform})
    return list_data


def load_map() -> list:
    door_list = []
    for d in DOORS:
        door_list.append((np.array([d[0], -d[1]]) * RESOLUTION) + ORIGIN)

    return door_list


def load_map_array(map_path: str) -> np.ndarray:
    map_array = np.array(Image.open(map_path))
    # Create meshgrid
    x = np.arange(0, map_array.shape[1]) * RESOLUTION + ORIGIN[0]
    y = -np.arange(0, map_array.shape[0]) * RESOLUTION + ORIGIN[1]
    xx, yy = np.meshgrid(x, y, indexing='xy')
    map_array = np.vstack((xx[None, ...], yy[None, ...], map_array[None, ...]))
    return map_array

def preprocess_mask(map_mask: np.ndarray, poses: np.ndarray) -> np.ndarray:
    # Scale mask
    scaled_mask = map_mask.copy()
    scaled_mask[2] /= 255.
    # Interpolate mask to get the respective mask at each pose by NN strategy
    interpolated_mask = np.zeros(poses.shape[0])
    for i, (x, y) in enumerate(poses[:, :2]):
        # Find closest pixel
        id_x = np.argmin(np.abs(map_mask[0, 0, :] - x))
        id_y = np.argmin(np.abs(map_mask[1, :, 0] - y))
        # Get value in mask
        value = scaled_mask[2, id_y, id_x]
        interpolated_mask[i] = value

    return interpolated_mask

if __name__ == '__main__':
    pass