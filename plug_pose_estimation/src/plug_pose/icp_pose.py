from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def run_point_to_point_icp(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_m: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, float]:
    result = o3d.pipelines.registration.registration_icp(
        model_pcd,
        observed_pcd,
        max_correspondence_m,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def transformed_copy(pcd: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    output = copy.deepcopy(pcd)
    output.transform(transform)
    return output
