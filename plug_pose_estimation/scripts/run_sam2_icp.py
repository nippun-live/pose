from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.icp_pose import run_icp_variant  # noqa: E402
from plug_pose.pointcloud import downsample_cloud  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.stl_utils import load_stl_as_pointcloud  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import colorize_depth, draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def load_mask_map(path: Path) -> dict[int, Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    output = {}
    for row in rows:
        mask_path = Path(row["mask"])
        if not mask_path.is_absolute():
            mask_path = base / mask_path.relative_to(path.parent.as_posix()) if row["mask"].startswith(path.parent.as_posix()) else mask_path
        if not mask_path.exists():
            mask_path = path.parent / "masks" / f"mask_{int(row['bag_frame_id']):06d}.png"
        output[int(row["bag_frame_id"])] = mask_path
    return output


def dominant_depth_mask(depth_m: np.ndarray, mask: np.ndarray, min_m: float, max_m: float, band_m: float) -> np.ndarray:
    candidate = (mask > 0) & (depth_m > min_m) & (depth_m < max_m)
    values = depth_m[candidate]
    if len(values) < 50:
        return candidate
    hist, bins = np.histogram(values, bins=80, range=(min_m, max_m))
    peak = float((bins[int(np.argmax(hist))] + bins[int(np.argmax(hist)) + 1]) / 2.0)
    return candidate & (depth_m >= peak - band_m) & (depth_m <= peak + band_m)


def backproject_mask(frame, mask_path: Path, intrinsics: dict, depth_scale: float, depth_cfg: dict, band_m: float):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.shape != frame.depth_z16.shape:
        mask = cv2.resize(mask, (frame.depth_z16.shape[1], frame.depth_z16.shape[0]), interpolation=cv2.INTER_NEAREST)

    depth_m = frame.depth_z16.astype(np.float64) * depth_scale
    keep = dominant_depth_mask(
        depth_m,
        mask,
        float(depth_cfg.get("min_m", 0.05)),
        float(depth_cfg.get("max_m", 0.5)),
        float(band_m),
    )
    vs, us = np.nonzero(keep)
    z = depth_m[vs, us]
    x = (us - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vs - intrinsics["ppy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z))
    colors = frame.color_rgb[vs, us].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, mask, keep.astype(np.uint8) * 255


def estimate_pca_pose(points: np.ndarray, stl_extent: np.ndarray, previous_pose: np.ndarray | None) -> np.ndarray:
    surface_center = np.median(points, axis=0)
    centered = points - surface_center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    x_axis = eigvecs[:, 0]
    z_axis = eigvecs[:, 2]
    to_camera = -surface_center
    to_camera /= np.linalg.norm(to_camera)
    if np.dot(z_axis, to_camera) < 0:
        z_axis = -z_axis
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    if previous_pose is not None:
        if np.dot(x_axis, previous_pose[:3, 0]) < 0:
            x_axis = -x_axis
    elif x_axis[0] > 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = surface_center - z_axis * (float(stl_extent[2]) / 2.0)
    return pose


def _apply_local_delta_limited(
    base_pose: np.ndarray,
    target_pose: np.ndarray,
    translation_blend: float,
    rotation_blend: float,
) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = base_pose[:3, 3] + translation_blend * (target_pose[:3, 3] - base_pose[:3, 3])
    delta_r = base_pose[:3, :3].T @ target_pose[:3, :3]
    delta_rotvec = Rotation.from_matrix(delta_r).as_rotvec()
    output[:3, :3] = base_pose[:3, :3] @ Rotation.from_rotvec(rotation_blend * delta_rotvec).as_matrix()
    return output


def pose_from_mask_and_previous(
    points: np.ndarray,
    stl_extent: np.ndarray,
    previous_pose: np.ndarray | None,
    init_mode: str = "previous_rotation",
    pca_rotation_blend: float = 0.35,
    translation_blend: float = 1.0,
) -> np.ndarray:
    if previous_pose is None:
        return estimate_pca_pose(points, stl_extent, None)
    if init_mode == "pca_each_frame":
        return estimate_pca_pose(points, stl_extent, previous_pose)

    local = (previous_pose[:3, :3].T @ (points - previous_pose[:3, 3]).T).T
    center_local = np.median(local, axis=0)
    center_local[2] = np.median(local[:, 2]) - float(stl_extent[2]) / 2.0
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = previous_pose[:3, :3]
    pose[:3, 3] = previous_pose[:3, 3] + previous_pose[:3, :3] @ center_local
    if translation_blend < 1.0:
        pose[:3, 3] = previous_pose[:3, 3] + translation_blend * (pose[:3, 3] - previous_pose[:3, 3])
    if init_mode == "previous_rotation":
        return pose
    if init_mode == "blended_pca":
        pca_pose = estimate_pca_pose(points, stl_extent, previous_pose)
        return _apply_local_delta_limited(pose, pca_pose, 0.0, pca_rotation_blend)
    if init_mode == "previous_pose":
        return previous_pose.copy()
    raise ValueError(f"Unsupported init mode: {init_mode}")


def relative_pose_step(base_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    translation_step = float(np.linalg.norm(target_pose[:3, 3] - base_pose[:3, 3]))
    rotation_step = rotation_error_deg(base_pose, target_pose)
    return translation_step, rotation_step


def gate_or_clamp_pose(
    init_pose: np.ndarray,
    icp_pose: np.ndarray,
    max_translation_step_m: float | None,
    max_rotation_step_deg: float | None,
    mode: str,
) -> tuple[np.ndarray, bool, float, float]:
    translation_step, rotation_step = relative_pose_step(init_pose, icp_pose)
    over_translation = max_translation_step_m is not None and translation_step > max_translation_step_m
    over_rotation = max_rotation_step_deg is not None and rotation_step > max_rotation_step_deg
    if not over_translation and not over_rotation:
        return icp_pose, False, translation_step, rotation_step
    if mode == "use_init":
        return init_pose.copy(), True, translation_step, rotation_step
    if mode == "clamp":
        translation_scale = 1.0
        if max_translation_step_m is not None and translation_step > 1e-12:
            translation_scale = min(translation_scale, max_translation_step_m / translation_step)
        rotation_scale = 1.0
        if max_rotation_step_deg is not None and rotation_step > 1e-12:
            rotation_scale = min(rotation_scale, max_rotation_step_deg / rotation_step)
        output = _apply_local_delta_limited(init_pose, icp_pose, translation_scale, rotation_scale)
        return output, True, translation_step, rotation_step
    raise ValueError(f"Unsupported motion gate mode: {mode}")
    return pose


def select_visible_model_points(
    model_pcd: o3d.geometry.PointCloud,
    approx_pose: np.ndarray,
    normal_dot_min: float,
    min_points: int,
) -> o3d.geometry.PointCloud:
    if not model_pcd.has_normals():
        model_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.004, max_nn=30)
        )
        model_pcd.normalize_normals()

    points_local = np.asarray(model_pcd.points)
    normals_local = np.asarray(model_pcd.normals)
    if len(points_local) == 0 or len(normals_local) != len(points_local):
        return model_pcd

    rotation = approx_pose[:3, :3]
    translation = approx_pose[:3, 3]
    points_cam = (rotation @ points_local.T).T + translation
    normals_cam = (rotation @ normals_local.T).T
    to_camera = -points_cam
    norms = np.linalg.norm(to_camera, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    to_camera[valid] /= norms[valid]
    visibility = np.sum(normals_cam * to_camera, axis=1)
    keep = valid & (visibility > normal_dot_min)
    if int(np.count_nonzero(keep)) < min_points:
        return model_pcd

    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(points_local[keep])
    if model_pcd.has_colors():
        output.colors = o3d.utility.Vector3dVector(np.asarray(model_pcd.colors)[keep])
    output.normals = o3d.utility.Vector3dVector(normals_local[keep])
    return output


def rotation_hypothesis_transform(base_pose: np.ndarray, angle_deg: float, axis: str) -> np.ndarray:
    output = base_pose.copy()
    rotation = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    if axis == "local_z":
        output[:3, :3] = base_pose[:3, :3] @ rotation
    elif axis == "camera_z":
        output[:3, :3] = rotation @ base_pose[:3, :3]
    else:
        raise ValueError(f"Unsupported hypothesis axis: {axis}")
    return output


def run_icp_with_hypotheses(
    model_pcd: o3d.geometry.PointCloud,
    observed_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    method: str,
    max_correspondence_m: float,
    max_iterations: int,
    normal_radius_m: float,
    robust_loss: str,
    robust_scale_m: float,
    hypothesis_step_deg: float,
    hypothesis_axis: str,
    rmse_weight: float,
) -> tuple[np.ndarray, float, float, float, int]:
    best = None
    angles = np.arange(0.0, 360.0, hypothesis_step_deg, dtype=np.float64)
    for angle_deg in angles:
        hypothesis_init = rotation_hypothesis_transform(init_transform, float(angle_deg), hypothesis_axis)
        transform, fitness, rmse = run_icp_variant(
            model_pcd,
            observed_pcd,
            hypothesis_init,
            method,
            max_correspondence_m,
            max_iterations,
            normal_radius_m=normal_radius_m,
            robust_loss=robust_loss,
            robust_scale_m=robust_scale_m,
        )
        score = float(fitness) - rmse_weight * float(rmse)
        if best is None or score > best[0]:
            best = (score, transform, fitness, rmse, float(angle_deg))
    if best is None:
        transform, fitness, rmse = run_icp_variant(
            model_pcd,
            observed_pcd,
            init_transform,
            method,
            max_correspondence_m,
            max_iterations,
            normal_radius_m=normal_radius_m,
            robust_loss=robust_loss,
            robust_scale_m=robust_scale_m,
        )
        return transform, fitness, rmse, 0.0, 1
    _score, transform, fitness, rmse, angle_deg = best
    return transform, fitness, rmse, angle_deg, len(angles)


def draw_overlay(frame, mask_depth, init_t, final_t, ref_t, intrinsics, stl_extent):
    image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    blue = np.zeros_like(image_bgr)
    blue[:, :, 0] = 255
    image_bgr = np.where(mask_depth[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.45, blue, 0.55, 0), image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    init_rvec, init_tvec = matrix_to_rvec_tvec(init_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, init_rvec, init_tvec, stl_extent, color=(0, 165, 255), thickness=2)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    final_rvec, final_tvec = matrix_to_rvec_tvec(final_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, final_rvec, final_tvec, stl_extent, color=(0, 255, 0), thickness=2)
    if ref_t is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ref_rvec, ref_tvec = matrix_to_rvec_tvec(ref_t)
        image_bgr = draw_projected_bbox(image_rgb, intrinsics, ref_rvec, ref_tvec, stl_extent, color=(255, 0, 255), thickness=2)
        text = "blue:SAM2+depth orange:init green:ICP magenta:ArUco ref"
    else:
        text = "blue:SAM2+depth orange:init green:ICP"
    cv2.putText(image_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image_bgr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--mask_map", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_csv", type=Path)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--depth_band_m", type=float, default=0.03)
    parser.add_argument("--min_observed_points", type=int, default=80)
    parser.add_argument("--lock_rotation", action="store_true")
    parser.add_argument(
        "--icp_method",
        choices=[
            "point_to_point",
            "point_to_plane",
            "robust_point_to_plane",
            "multiscale_point_to_point",
            "multiscale_point_to_plane",
            "multiscale_robust_point_to_plane",
        ],
        default="point_to_point",
    )
    parser.add_argument("--normal_radius_m", type=float, default=0.006)
    parser.add_argument("--robust_loss", choices=["huber", "tukey"], default="huber")
    parser.add_argument("--robust_scale_m", type=float, default=0.006)
    parser.add_argument("--visible_stl", action="store_true")
    parser.add_argument("--visible_normal_dot_min", type=float, default=0.0)
    parser.add_argument("--min_visible_model_points", type=int, default=300)
    parser.add_argument("--multi_hypothesis", action="store_true")
    parser.add_argument("--hypothesis_step_deg", type=float, default=45.0)
    parser.add_argument("--hypothesis_axis", choices=["local_z", "camera_z"], default="local_z")
    parser.add_argument("--hypothesis_rmse_weight", type=float, default=25.0)
    parser.add_argument(
        "--init_mode",
        choices=["previous_rotation", "pca_each_frame", "blended_pca", "previous_pose"],
        default="previous_rotation",
    )
    parser.add_argument("--pca_rotation_blend", type=float, default=0.35)
    parser.add_argument("--translation_blend", type=float, default=1.0)
    parser.add_argument("--max_icp_translation_step_m", type=float)
    parser.add_argument("--max_icp_rotation_step_deg", type=float)
    parser.add_argument("--motion_gate_mode", choices=["clamp", "use_init"], default="clamp")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    depth_cfg = config["markerless"]["depth"]
    icp_cfg = config["markerless"]["icp"]
    mask_paths = load_mask_map(args.mask_map)
    ref_rows = load_pose_csv(args.reference_csv) if args.reference_csv else {}

    model_full = load_stl_as_pointcloud(args.stl, int(icp_cfg["n_model_points"]))
    model = downsample_cloud(model_full, float(icp_cfg["voxel_size_m"]))
    stl_extent = get_projectable_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()
    if depth_scale is None:
        raise RuntimeError("No depth scale found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    depth_mask_dir = args.out_dir / "depth_masks"
    cloud_dir = args.out_dir / "clouds"
    for directory in [overlay_dir, depth_mask_dir, cloud_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "poses_sam2_icp.csv"
    out_video = args.out_dir / "sam2_icp_overlay.mp4"

    previous_pose = None
    rows = []
    writer = None
    end_frame = None if args.max_frames is None else args.start_frame + args.max_frames
    for frame in iter_aligned_frames(args.bag, max_frames=end_frame):
        if frame.frame_id < args.start_frame:
            continue
        mask_path = mask_paths.get(frame.frame_id)
        if mask_path is None:
            continue

        ref_t = None
        ref_row = ref_rows.get(frame.frame_id)
        if ref_row is not None and ref_row["detected"] == "1":
            ref_t = row_to_transform(ref_row)

        try:
            observed_raw, sam_mask, depth_mask = backproject_mask(
                frame,
                mask_path,
                intrinsics,
                depth_scale,
                depth_cfg,
                args.depth_band_m,
            )
        except FileNotFoundError:
            continue
        observed = downsample_cloud(observed_raw, float(icp_cfg["voxel_size_m"]))
        if len(observed.points) < int(args.min_observed_points):
            rows.append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": f"{frame.timestamp_s:.6f}",
                    "detected": 0,
                    "icp_method": args.icp_method,
                    "visible_stl": int(args.visible_stl),
                    "multi_hypothesis": int(args.multi_hypothesis),
                    "init_mode": args.init_mode,
                    "motion_gate_mode": args.motion_gate_mode,
                    "motion_gate_triggered": "",
                    "icp_translation_step_m": "",
                    "icp_rotation_step_deg": "",
                    "selected_hypothesis_deg": "",
                    "hypothesis_count": "",
                    "observed_points": len(observed.points),
                    "model_points": "",
                    "fitness": "",
                    "rmse": "",
                    "tx": "",
                    "ty": "",
                    "tz": "",
                    "qx": "",
                    "qy": "",
                    "qz": "",
                    "qw": "",
                    "ref_translation_error_m": "",
                    "ref_rotation_error_deg": "",
                }
            )
            continue

        observed_for_icp = observed
        model_for_icp = model
        if args.icp_method.startswith("multiscale_"):
            observed_for_icp = observed_raw
            model_for_icp = model_full

        points = np.asarray(observed.points)
        init_t = pose_from_mask_and_previous(
            points,
            stl_extent,
            previous_pose,
            init_mode=args.init_mode,
            pca_rotation_blend=args.pca_rotation_blend,
            translation_blend=args.translation_blend,
        )
        if args.visible_stl:
            model_for_icp = select_visible_model_points(
                model_for_icp,
                init_t,
                args.visible_normal_dot_min,
                args.min_visible_model_points,
            )

        selected_hypothesis_deg = ""
        hypothesis_count = ""
        if args.multi_hypothesis:
            icp_t, fitness, rmse, selected_hypothesis_deg_value, hypothesis_count_value = run_icp_with_hypotheses(
                model_for_icp,
                observed_for_icp,
                init_t,
                args.icp_method,
                float(icp_cfg["max_correspondence_m"]),
                int(icp_cfg["max_iterations"]),
                args.normal_radius_m,
                args.robust_loss,
                args.robust_scale_m,
                args.hypothesis_step_deg,
                args.hypothesis_axis,
                args.hypothesis_rmse_weight,
            )
            selected_hypothesis_deg = f"{selected_hypothesis_deg_value:.6f}"
            hypothesis_count = str(hypothesis_count_value)
        else:
            icp_t, fitness, rmse = run_icp_variant(
                model_for_icp,
                observed_for_icp,
                init_t,
                args.icp_method,
                float(icp_cfg["max_correspondence_m"]),
                int(icp_cfg["max_iterations"]),
                normal_radius_m=args.normal_radius_m,
                robust_loss=args.robust_loss,
                robust_scale_m=args.robust_scale_m,
            )
        if args.lock_rotation:
            locked = np.eye(4, dtype=np.float64)
            locked[:3, :3] = init_t[:3, :3]
            locked[:3, 3] = icp_t[:3, 3]
            icp_t = locked
            eval_result = o3d.pipelines.registration.evaluate_registration(
                model,
                observed,
                float(icp_cfg["max_correspondence_m"]),
                icp_t,
            )
            fitness = float(eval_result.fitness)
            rmse = float(eval_result.inlier_rmse)

        icp_t, motion_gate_triggered, icp_translation_step, icp_rotation_step = gate_or_clamp_pose(
            init_t,
            icp_t,
            args.max_icp_translation_step_m,
            args.max_icp_rotation_step_deg,
            args.motion_gate_mode,
        )
        if motion_gate_triggered:
            eval_result = o3d.pipelines.registration.evaluate_registration(
                model,
                observed,
                float(icp_cfg["max_correspondence_m"]),
                icp_t,
            )
            fitness = float(eval_result.fitness)
            rmse = float(eval_result.inlier_rmse)

        previous_pose = icp_t
        quat = matrix_to_quaternion_xyzw(icp_t)
        ref_translation_error = ""
        ref_rotation_error = ""
        if ref_t is not None:
            ref_translation_error = f"{np.linalg.norm(icp_t[:3, 3] - ref_t[:3, 3]):.9f}"
            ref_rotation_error = f"{rotation_error_deg(ref_t, icp_t):.6f}"

        rows.append(
            {
                "frame_id": frame.frame_id,
                "timestamp": f"{frame.timestamp_s:.6f}",
                "detected": 1,
                "icp_method": args.icp_method,
                "visible_stl": int(args.visible_stl),
                "multi_hypothesis": int(args.multi_hypothesis),
                "init_mode": args.init_mode,
                "motion_gate_mode": args.motion_gate_mode,
                "motion_gate_triggered": int(motion_gate_triggered),
                "icp_translation_step_m": f"{icp_translation_step:.9f}",
                "icp_rotation_step_deg": f"{icp_rotation_step:.6f}",
                "selected_hypothesis_deg": selected_hypothesis_deg,
                "hypothesis_count": hypothesis_count,
                "observed_points": len(observed.points),
                "model_points": len(model_for_icp.points),
                "fitness": f"{fitness:.9f}",
                "rmse": f"{rmse:.9f}",
                "tx": f"{icp_t[0, 3]:.9f}",
                "ty": f"{icp_t[1, 3]:.9f}",
                "tz": f"{icp_t[2, 3]:.9f}",
                "qx": f"{quat[0]:.9f}",
                "qy": f"{quat[1]:.9f}",
                "qz": f"{quat[2]:.9f}",
                "qw": f"{quat[3]:.9f}",
                "ref_translation_error_m": ref_translation_error,
                "ref_rotation_error_deg": ref_rotation_error,
            }
        )

        overlay = draw_overlay(frame, depth_mask, init_t, icp_t, ref_t, intrinsics, stl_extent)
        cv2.imwrite(str(overlay_dir / f"frame_{frame.frame_id:06d}_overlay.png"), overlay)
        cv2.imwrite(str(depth_mask_dir / f"frame_{frame.frame_id:06d}_depth_mask.png"), depth_mask)
        if frame.frame_id % 20 == 0:
            o3d.io.write_point_cloud(str(cloud_dir / f"frame_{frame.frame_id:06d}_observed.ply"), observed)

        if writer is None:
            h, w = overlay.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (w, h))
        writer.write(overlay)

    if writer is not None:
        writer.release()

    fieldnames = [
        "frame_id",
        "timestamp",
        "detected",
        "icp_method",
        "visible_stl",
        "multi_hypothesis",
        "init_mode",
        "motion_gate_mode",
        "motion_gate_triggered",
        "icp_translation_step_m",
        "icp_rotation_step_deg",
        "selected_hypothesis_deg",
        "hypothesis_count",
        "observed_points",
        "model_points",
        "fitness",
        "rmse",
        "tx",
        "ty",
        "tz",
        "qx",
        "qy",
        "qz",
        "qw",
        "ref_translation_error_m",
        "ref_rotation_error_deg",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"saved {len(rows)} pose rows to {out_csv}")
    print(f"saved overlay video to {out_video}")


if __name__ == "__main__":
    main()
