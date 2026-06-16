from __future__ import annotations

import cv2
import numpy as np

from plug_pose.bag_reader import intrinsics_to_camera_matrix, intrinsics_to_dist_coeffs


def make_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_marker(
    image_rgb: np.ndarray,
    marker_id: int = 32,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    detector = make_detector()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None, None

    ids_flat = ids.flatten()
    for idx, detected_id in enumerate(ids_flat):
        if int(detected_id) == marker_id:
            return corners[idx], np.array([[detected_id]], dtype=np.int32)
    return None, None


def estimate_marker_pose(
    corners: np.ndarray,
    intrinsics: dict,
    marker_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = intrinsics_to_camera_matrix(intrinsics)
    dist_coeffs = intrinsics_to_dist_coeffs(intrinsics)
    rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
        [corners], marker_length_m, camera_matrix, dist_coeffs
    )
    return rvecs[0].reshape(3), tvecs[0].reshape(3)

