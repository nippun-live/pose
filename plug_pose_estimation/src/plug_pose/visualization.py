from __future__ import annotations

import cv2
import numpy as np

from plug_pose.bag_reader import intrinsics_to_camera_matrix, intrinsics_to_dist_coeffs


def colorize_depth(depth_z16: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_z16, 0, np.percentile(depth_z16[depth_z16 > 0], 98) if np.any(depth_z16 > 0) else 1)
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def draw_marker_detection(image_rgb: np.ndarray, corners: np.ndarray | None, ids: np.ndarray | None) -> np.ndarray:
    output_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(output_bgr, [corners], ids)
    return output_bgr


def draw_pose_axes(
    image_rgb: np.ndarray,
    intrinsics: dict,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length_m: float = 0.01,
) -> np.ndarray:
    output_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.drawFrameAxes(
        output_bgr,
        intrinsics_to_camera_matrix(intrinsics),
        intrinsics_to_dist_coeffs(intrinsics),
        rvec,
        tvec,
        axis_length_m,
    )
    return output_bgr


def draw_projected_bbox(
    image_rgb: np.ndarray,
    intrinsics: dict,
    rvec: np.ndarray,
    tvec: np.ndarray,
    extent_xyz_m: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    output_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ex, ey, ez = np.asarray(extent_xyz_m, dtype=np.float32) / 2.0
    corners = np.array(
        [
            [-ex, -ey, -ez],
            [ex, -ey, -ez],
            [ex, ey, -ez],
            [-ex, ey, -ez],
            [-ex, -ey, ez],
            [ex, -ey, ez],
            [ex, ey, ez],
            [-ex, ey, ez],
        ],
        dtype=np.float32,
    )
    points, _ = cv2.projectPoints(
        corners,
        rvec,
        tvec,
        intrinsics_to_camera_matrix(intrinsics),
        intrinsics_to_dist_coeffs(intrinsics),
    )
    points = np.round(points.reshape(-1, 2)).astype(int)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        cv2.line(output_bgr, tuple(points[start]), tuple(points[end]), color, thickness, cv2.LINE_AA)
    return output_bgr


def draw_projected_wireframe(
    image_rgb: np.ndarray,
    intrinsics: dict,
    rvec: np.ndarray,
    tvec: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    output_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    camera_matrix = intrinsics_to_camera_matrix(intrinsics)
    dist_coeffs = intrinsics_to_dist_coeffs(intrinsics)

    points_2d, _ = cv2.projectPoints(
        np.asarray(vertices, dtype=np.float32),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    points_2d = np.round(points_2d.reshape(-1, 2)).astype(np.int32)

    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ np.asarray(vertices, dtype=np.float64).T).T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    valid_depth = camera_points[:, 2] > 0.01

    height, width = image_rgb.shape[:2]
    margin = max(width, height) * 2
    for start, end in np.asarray(edges, dtype=np.int32):
        if not (valid_depth[start] and valid_depth[end]):
            continue
        p0 = points_2d[start]
        p1 = points_2d[end]
        if (
            min(p0[0], p1[0]) < -margin
            or max(p0[0], p1[0]) > width + margin
            or min(p0[1], p1[1]) < -margin
            or max(p0[1], p1[1]) > height + margin
        ):
            continue
        cv2.line(output_bgr, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)
    return output_bgr
