from __future__ import annotations

import numpy as np
import open3d as o3d


def backproject_depth_roi(
    depth_z16: np.ndarray,
    color_rgb: np.ndarray,
    intrinsics: dict,
    depth_scale_m: float,
    roi: dict,
    min_depth_m: float,
    max_depth_m: float,
) -> o3d.geometry.PointCloud:
    x_min = max(0, int(roi["x_min"]))
    y_min = max(0, int(roi["y_min"]))
    x_max = min(depth_z16.shape[1], int(roi["x_max"]))
    y_max = min(depth_z16.shape[0], int(roi["y_max"]))

    depth_roi = depth_z16[y_min:y_max, x_min:x_max].astype(np.float64) * depth_scale_m
    valid = (depth_roi > min_depth_m) & (depth_roi < max_depth_m)
    if not np.any(valid):
        return o3d.geometry.PointCloud()

    ys, xs = np.nonzero(valid)
    z = depth_roi[ys, xs]
    u = xs + x_min
    v = ys + y_min

    x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z))

    colors = color_rgb[v, u].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def downsample_cloud(pcd: o3d.geometry.PointCloud, voxel_size_m: float) -> o3d.geometry.PointCloud:
    if pcd.is_empty() or voxel_size_m <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size_m)


def estimate_normals(pcd: o3d.geometry.PointCloud, radius_m: float) -> None:
    if pcd.is_empty():
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_m, max_nn=30)
    )


def filter_by_pose_bbox(
    pcd: o3d.geometry.PointCloud,
    transform: np.ndarray,
    extent_xyz_m: np.ndarray,
    margin_m: float,
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    local = (rotation.T @ (points - translation).T).T
    half_extent = np.asarray(extent_xyz_m, dtype=np.float64) / 2.0 + float(margin_m)
    keep = np.all(np.abs(local) <= half_extent, axis=1)

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[keep])
    if colors is not None:
        filtered.colors = o3d.utility.Vector3dVector(colors[keep])
    return filtered
