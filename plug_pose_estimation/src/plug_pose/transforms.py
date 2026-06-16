from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(translation: np.ndarray, rotation: Rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def rpy_deg_to_matrix(translation_m: list[float], rotation_rpy_deg: list[float]) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", rotation_rpy_deg, degrees=True)
    return make_transform(np.asarray(translation_m, dtype=np.float64), rotation)


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return matrix


def pose_quat_xyzw_to_matrix(translation: list[float], quaternion_xyzw: list[float]) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4))
    return make_transform(np.asarray(translation, dtype=np.float64), rotation)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix[:3, :3]).as_quat()


def matrix_to_rvec_tvec(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
    tvec = matrix[:3, 3]
    return rvec.reshape(3), tvec.reshape(3)
