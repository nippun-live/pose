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


def _copy_downsampled_with_normals(
    pcd: o3d.geometry.PointCloud,
    voxel_size_m: float | None,
    normal_radius_m: float,
) -> o3d.geometry.PointCloud:
    output = copy.deepcopy(pcd)
    if voxel_size_m is not None and voxel_size_m > 0:
        output = output.voxel_down_sample(voxel_size_m)
    if len(output.points) > 0:
        output.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius_m,
                max_nn=30,
            )
        )
        output.normalize_normals()
    return output


def _robust_kernel(name: str, scale_m: float):
    name = name.lower()
    if name == "huber":
        return o3d.pipelines.registration.HuberLoss(scale_m)
    if name == "tukey":
        return o3d.pipelines.registration.TukeyLoss(scale_m)
    raise ValueError(f"Unsupported robust loss: {name}")


def run_point_to_plane_icp(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_m: float,
    max_iterations: int,
    normal_radius_m: float,
) -> tuple[np.ndarray, float, float]:
    model = _copy_downsampled_with_normals(model_pcd, None, normal_radius_m)
    observed = _copy_downsampled_with_normals(observed_pcd, None, normal_radius_m)
    result = o3d.pipelines.registration.registration_icp(
        model,
        observed,
        max_correspondence_m,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def run_robust_point_to_plane_icp(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_m: float,
    max_iterations: int,
    normal_radius_m: float,
    robust_loss: str,
    robust_scale_m: float,
) -> tuple[np.ndarray, float, float]:
    model = _copy_downsampled_with_normals(model_pcd, None, normal_radius_m)
    observed = _copy_downsampled_with_normals(observed_pcd, None, normal_radius_m)
    result = o3d.pipelines.registration.registration_icp(
        model,
        observed,
        max_correspondence_m,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            _robust_kernel(robust_loss, robust_scale_m)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def run_multiscale_icp(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    stages: list[tuple[float, float, int]],
    method: str,
    normal_radius_m: float,
    robust_loss: str,
    robust_scale_m: float,
) -> tuple[np.ndarray, float, float]:
    transform = init_transform
    fitness = 0.0
    rmse = 0.0
    for voxel_size_m, max_correspondence_m, max_iterations in stages:
        model = _copy_downsampled_with_normals(model_pcd, voxel_size_m, normal_radius_m)
        observed = _copy_downsampled_with_normals(observed_pcd, voxel_size_m, normal_radius_m)
        if len(model.points) == 0 or len(observed.points) == 0:
            continue
        if method == "point_to_point":
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif method == "point_to_plane":
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        elif method == "robust_point_to_plane":
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                _robust_kernel(robust_loss, robust_scale_m)
            )
        else:
            raise ValueError(f"Unsupported multiscale ICP method: {method}")
        result = o3d.pipelines.registration.registration_icp(
            model,
            observed,
            max_correspondence_m,
            transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
        )
        transform = result.transformation
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)
    return transform, fitness, rmse


def run_icp_variant(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    method: str,
    max_correspondence_m: float,
    max_iterations: int,
    normal_radius_m: float = 0.006,
    robust_loss: str = "huber",
    robust_scale_m: float = 0.006,
) -> tuple[np.ndarray, float, float]:
    if method == "point_to_point":
        return run_point_to_point_icp(
            model_pcd,
            observed_pcd,
            init_transform,
            max_correspondence_m,
            max_iterations,
        )
    if method == "point_to_plane":
        return run_point_to_plane_icp(
            model_pcd,
            observed_pcd,
            init_transform,
            max_correspondence_m,
            max_iterations,
            normal_radius_m,
        )
    if method == "robust_point_to_plane":
        return run_robust_point_to_plane_icp(
            model_pcd,
            observed_pcd,
            init_transform,
            max_correspondence_m,
            max_iterations,
            normal_radius_m,
            robust_loss,
            robust_scale_m,
        )
    if method.startswith("multiscale_"):
        base_method = method.removeprefix("multiscale_")
        stages = [
            (0.0050, 0.0150, max(15, max_iterations // 2)),
            (0.0030, 0.0090, max(20, max_iterations // 2)),
            (0.0015, max_correspondence_m, max_iterations),
        ]
        return run_multiscale_icp(
            model_pcd,
            observed_pcd,
            init_transform,
            stages,
            base_method,
            normal_radius_m,
            robust_loss,
            robust_scale_m,
        )
    raise ValueError(f"Unsupported ICP method: {method}")


def transformed_copy(pcd: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    output = copy.deepcopy(pcd)
    output.transform(transform)
    return output
