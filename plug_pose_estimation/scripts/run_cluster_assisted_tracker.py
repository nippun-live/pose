from __future__ import annotations

import argparse
import csv
import math
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
from plug_pose.icp_pose import run_point_to_point_icp  # noqa: E402
from plug_pose.pointcloud import downsample_cloud  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.stl_utils import load_stl_as_pointcloud  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import colorize_depth, draw_projected_bbox  # noqa: E402


PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
    ],
    dtype=np.uint8,
)


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


def resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def backproject_candidates(frame, intrinsics, depth_scale, roi, depth_cfg, color_cfg):
    x_min = max(0, int(roi["x_min"]))
    y_min = max(0, int(roi["y_min"]))
    x_max = min(frame.depth_z16.shape[1], int(roi["x_max"]))
    y_max = min(frame.depth_z16.shape[0], int(roi["y_max"]))

    depth_m = frame.depth_z16.astype(np.float64) * depth_scale
    roi_depth = depth_m[y_min:y_max, x_min:x_max]
    valid = (roi_depth > depth_cfg["min_m"]) & (roi_depth < depth_cfg["max_m"])

    if color_cfg and color_cfg.get("enabled", False):
        hsv = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[y_min:y_max, x_min:x_max, 1]
        val = hsv[y_min:y_max, x_min:x_max, 2]
        valid &= (sat <= int(color_cfg["saturation_max"])) & (val >= int(color_cfg["value_min"]))

    ys_roi, xs_roi = np.nonzero(valid)
    us = xs_roi + x_min
    vs = ys_roi + y_min
    z = depth_m[vs, us]
    x = (us - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vs - intrinsics["ppy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z))
    colors = frame.color_rgb[vs, us].astype(np.float64) / 255.0
    pixels = np.column_stack((us, vs))
    return points, colors, pixels, (x_min, y_min, x_max, y_max)


def dimension_score(extent, cluster_cfg):
    target = np.array([0.05645, 0.01469, 0.01185])
    dims = np.sort(np.asarray(extent, dtype=np.float64))[::-1]
    relative = np.abs(dims - target) / target
    min_long = float(cluster_cfg.get("min_long_extent_m", 0.0))
    max_long = float(cluster_cfg.get("max_long_extent_m", 999.0))
    long_penalty = 0.0
    if dims[0] < min_long:
        long_penalty += (min_long - dims[0]) * 80.0
    if dims[0] > max_long:
        long_penalty += (dims[0] - max_long) * 80.0
    return float(relative[0] * 4.0 + relative[1] + relative[2] + long_penalty)


def extent_rejection_reason(extent, cluster_cfg) -> str | None:
    dims = np.sort(np.asarray(extent, dtype=np.float64))[::-1]
    checks = [
        ("long", dims[0], cluster_cfg.get("reject_long_extent_m")),
        ("mid", dims[1], cluster_cfg.get("reject_mid_extent_m")),
        ("short", dims[2], cluster_cfg.get("reject_short_extent_m")),
    ]
    for name, value, limit in checks:
        if limit is not None and value > float(limit):
            return f"{name}_extent>{float(limit):.4f}"
    return None


def filter_by_previous_pose_gate(
    points: np.ndarray,
    colors: np.ndarray,
    pixels: np.ndarray,
    previous_pose: np.ndarray | None,
    stl_extent: np.ndarray,
    margin_m: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]:
    if previous_pose is None:
        return points, colors, pixels, None, False

    local = (previous_pose[:3, :3].T @ (points - previous_pose[:3, 3]).T).T
    half_extent = np.asarray(stl_extent, dtype=np.float64) / 2.0 + float(margin_m)
    keep = np.all(np.abs(local) <= half_extent, axis=1)

    # If the gate is too tight for a bad previous pose, keep the broad candidates
    # so the tracker can recover instead of dropping the frame entirely.
    if int(np.count_nonzero(keep)) < min_points:
        return points, colors, pixels, keep, False

    return points[keep], colors[keep], pixels[keep], keep, True


def estimate_cluster_pca_pose(points: np.ndarray, stl_extent: np.ndarray, previous_pose: np.ndarray | None) -> np.ndarray:
    surface_center = points.mean(axis=0)
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

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = surface_center - z_axis * (float(stl_extent[2]) / 2.0)
    return transform


def select_cluster(points, colors, pixels, cluster_cfg, stl_extent, previous_pose, previous_mask):
    if len(points) == 0:
        return None, np.array([], dtype=np.int32), []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(cluster_cfg["dbscan_eps_m"]),
            min_points=int(cluster_cfg["dbscan_min_points"]),
            print_progress=False,
        )
    )

    candidates = []
    min_cluster_points = int(cluster_cfg["min_cluster_points"])
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if len(idx) < min_cluster_points:
            continue
        cluster = pcd.select_by_index(idx.tolist())
        aabb = cluster.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        reject_reason = extent_rejection_reason(extent, cluster_cfg)
        if reject_reason is not None:
            continue
        center = aabb.get_center()
        try:
            init_pose = estimate_cluster_pca_pose(points[idx], stl_extent, previous_pose)
        except Exception:
            continue

        score = dimension_score(extent, cluster_cfg)
        temporal_distance = ""
        temporal_distance_value = None
        overlap_ratio = None
        if previous_pose is not None:
            distance = float(np.linalg.norm(init_pose[:3, 3] - previous_pose[:3, 3]))
            temporal_distance_value = distance
            temporal_distance = f"{distance:.6f}"
            score += float(cluster_cfg.get("temporal_weight", 1.0)) * distance / 0.03

        if previous_mask is not None:
            cluster_pixels = pixels[idx]
            inside = previous_mask[cluster_pixels[:, 1], cluster_pixels[:, 0]] > 0
            overlap_ratio = float(np.count_nonzero(inside) / max(1, len(cluster_pixels)))
            score += float(cluster_cfg.get("previous_mask_overlap_weight", 0.0)) * (1.0 - overlap_ratio)

        candidates.append(
            {
                "label": label,
                "idx": idx,
                "points": len(idx),
                "center": center,
                "extent": extent,
                "pose": init_pose,
                "score": score,
                "temporal_distance": temporal_distance,
                "temporal_distance_value": temporal_distance_value,
                "previous_mask_overlap": overlap_ratio,
            }
        )

    if not candidates:
        return None, labels, []
    if previous_mask is not None and "previous_mask_overlap_min" in cluster_cfg:
        min_overlap = float(cluster_cfg["previous_mask_overlap_min"])
        overlapping = [
            item
            for item in candidates
            if item["previous_mask_overlap"] is not None and item["previous_mask_overlap"] >= min_overlap
        ]
        if overlapping:
            candidates = overlapping
    if previous_pose is not None and "temporal_max_distance_m" in cluster_cfg:
        max_distance = float(cluster_cfg["temporal_max_distance_m"])
        nearby = [
            item
            for item in candidates
            if item["temporal_distance_value"] is not None and item["temporal_distance_value"] <= max_distance
        ]
        if nearby:
            candidates = nearby
    candidates.sort(key=lambda item: item["score"])
    return candidates[0], labels, candidates


def pose_from_cluster_translation_and_previous_orientation(
    cluster_pose: np.ndarray,
    selected_points: np.ndarray,
    stl_extent: np.ndarray,
    previous_pose: np.ndarray | None,
    use_previous_orientation: bool,
) -> np.ndarray:
    if previous_pose is None or not use_previous_orientation:
        return cluster_pose

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = previous_pose[:3, :3]

    # Keep the previous orientation, but recenter using the new observed surface.
    # Project selected points into the previous plug frame. The visible face is
    # near +Z, so the plug center is half thickness behind that face.
    local = (previous_pose[:3, :3].T @ (selected_points - previous_pose[:3, 3]).T).T
    center_local = np.median(local, axis=0)
    center_local[2] = np.median(local[:, 2]) - float(stl_extent[2]) / 2.0
    transform[:3, 3] = previous_pose[:3, 3] + previous_pose[:3, :3] @ center_local
    return transform


def evaluate_alignment(model, observed, transform, max_correspondence_m):
    result = o3d.pipelines.registration.evaluate_registration(
        model,
        observed,
        max_correspondence_m,
        transform,
    )
    return float(result.fitness), float(result.inlier_rmse)


def choose_icp_pose(init_t, icp_t, init_fitness, init_rmse, icp_fitness, icp_rmse, icp_cfg):
    acceptance_cfg = icp_cfg.get("acceptance", {})
    if not acceptance_cfg.get("enabled", False):
        return icp_t, True, ""

    max_translation = float(acceptance_cfg.get("max_translation_step_m", 999.0))
    min_rmse_improvement = float(acceptance_cfg.get("min_rmse_improvement_m", 0.0))
    min_fitness_improvement = float(acceptance_cfg.get("min_fitness_improvement", 0.0))

    translation_step = float(np.linalg.norm(icp_t[:3, 3] - init_t[:3, 3]))
    rmse_improvement = init_rmse - icp_rmse
    fitness_improvement = icp_fitness - init_fitness

    reasons = []
    if translation_step > max_translation:
        reasons.append(f"translation_step>{max_translation:.4f}")
    if rmse_improvement < min_rmse_improvement and fitness_improvement < min_fitness_improvement:
        reasons.append("weak_fit_improvement")

    if reasons:
        return init_t, False, ";".join(reasons)
    return icp_t, True, ""


def bound_rotation_update(init_t, icp_t, rotation_gate_cfg):
    if not rotation_gate_cfg.get("enabled", False):
        return icp_t, rotation_error_deg(init_t, icp_t), False

    max_step_deg = float(rotation_gate_cfg.get("max_step_deg", 180.0))
    mode = rotation_gate_cfg.get("mode", "clamp")
    rotation_step = rotation_error_deg(init_t, icp_t)
    if rotation_step <= max_step_deg:
        return icp_t, rotation_step, False

    bounded = icp_t.copy()
    if mode == "keep":
        bounded[:3, :3] = init_t[:3, :3]
    elif mode == "clamp":
        delta = init_t[:3, :3].T @ icp_t[:3, :3]
        delta_rotvec = Rotation.from_matrix(delta).as_rotvec()
        delta_angle = float(np.linalg.norm(delta_rotvec))
        if delta_angle > 1e-12:
            scale = np.radians(max_step_deg) / delta_angle
            bounded_rotation = Rotation.from_matrix(init_t[:3, :3]) * Rotation.from_rotvec(delta_rotvec * scale)
            bounded[:3, :3] = bounded_rotation.as_matrix()
    else:
        raise ValueError(f"Unknown rotation gate mode: {mode}")
    return bounded, rotation_step, True


def save_cluster_summary(path: Path, candidates, selected_label) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "selected",
            "label",
            "points",
            "center_x",
            "center_y",
            "center_z",
            "extent_x",
            "extent_y",
            "extent_z",
            "extent_sorted_desc",
            "temporal_distance_m",
            "previous_mask_overlap",
            "score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            extent = item["extent"]
            center = item["center"]
            writer.writerow(
                {
                    "selected": int(item["label"] == selected_label),
                    "label": item["label"],
                    "points": item["points"],
                    "center_x": f"{center[0]:.6f}",
                    "center_y": f"{center[1]:.6f}",
                    "center_z": f"{center[2]:.6f}",
                    "extent_x": f"{extent[0]:.6f}",
                    "extent_y": f"{extent[1]:.6f}",
                    "extent_z": f"{extent[2]:.6f}",
                    "extent_sorted_desc": " ".join(f"{v:.6f}" for v in np.sort(extent)[::-1]),
                    "temporal_distance_m": item["temporal_distance"],
                    "previous_mask_overlap": ""
                    if item["previous_mask_overlap"] is None
                    else f"{item['previous_mask_overlap']:.6f}",
                    "score": f"{item['score']:.6f}",
                }
            )


def draw_frame_outputs(
    frame,
    mask,
    init_t,
    icp_t,
    ref_t,
    intrinsics,
    stl_extent,
    roi_box,
    label,
):
    color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    depth_bgr = colorize_depth(frame.depth_z16)
    color_small = resize_keep_aspect(color_bgr, 320)
    depth_small = resize_keep_aspect(depth_bgr, 320)
    color_depth = np.hstack([color_small, depth_small])
    cv2.putText(color_depth, f"frame {frame.frame_id:06d}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    mask_overlay = color_bgr.copy()
    blue = np.zeros_like(mask_overlay)
    blue[:, :, 0] = 255
    masked_color = np.where(mask[:, :, None] > 0, cv2.addWeighted(mask_overlay, 0.45, blue, 0.55, 0), mask_overlay)
    mask_overlay = masked_color.copy()
    x_min, y_min, x_max, y_max = roi_box
    cv2.rectangle(mask_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    cv2.putText(mask_overlay, f"selected cluster {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    bbox_bgr = masked_color.copy()
    cv2.rectangle(bbox_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    init_rvec, init_tvec = matrix_to_rvec_tvec(init_t)
    bbox_bgr = draw_projected_bbox(
        cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB),
        intrinsics,
        init_rvec,
        init_tvec,
        stl_extent,
        color=(0, 165, 255),
        thickness=2,
    )
    icp_rvec, icp_tvec = matrix_to_rvec_tvec(icp_t)
    bbox_bgr = draw_projected_bbox(
        cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB),
        intrinsics,
        icp_rvec,
        icp_tvec,
        stl_extent,
        color=(0, 255, 0),
        thickness=2,
    )
    if ref_t is not None:
        ref_rvec, ref_tvec = matrix_to_rvec_tvec(ref_t)
        bbox_bgr = draw_projected_bbox(
            cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB),
            intrinsics,
            ref_rvec,
            ref_tvec,
            stl_extent,
            color=(255, 0, 255),
            thickness=2,
        )
        text = "blue:mask orange:cluster init green:ICP magenta:ArUco ref"
    else:
        text = "blue:mask orange:cluster init green:ICP"
    cv2.putText(bbox_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bbox_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return color_depth, mask_overlay, bbox_bgr


def draw_gate_overlay(frame, gate_mask, roi_box, gate_used: bool):
    output = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    if gate_mask is not None:
        cyan = np.zeros_like(output)
        cyan[:, :, 0] = 255
        cyan[:, :, 1] = 255
        output = np.where(gate_mask[:, :, None] > 0, cv2.addWeighted(output, 0.45, cyan, 0.55, 0), output)
    x_min, y_min, x_max, y_max = roi_box
    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    status = "used" if gate_used else "not used"
    cv2.putText(output, f"pose gate {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--roi_key", default="cluster_roi_tagged")
    parser.add_argument("--reference_csv", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=127)
    parser.add_argument("--max_frames", type=int, default=44)
    parser.add_argument("--lock_rotation", action="store_true")
    parser.add_argument("--disable_icp_acceptance", action="store_true")
    parser.add_argument("--bounded_rotation_deg", type=float)
    parser.add_argument("--rotation_gate_mode", choices=["clamp", "keep"])
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    markerless = config["markerless"]
    roi = markerless[args.roi_key]
    depth_cfg = markerless["depth"]
    cluster_cfg = markerless["clustering"]
    color_cfg = cluster_cfg.get("color_filter", {})
    icp_cfg = markerless["icp"]
    if args.disable_icp_acceptance:
        icp_cfg = dict(icp_cfg)
        icp_cfg["acceptance"] = {"enabled": False}
    if args.bounded_rotation_deg is not None:
        icp_cfg = dict(icp_cfg)
        icp_cfg["rotation_gate"] = {
            "enabled": True,
            "max_step_deg": args.bounded_rotation_deg,
            "mode": args.rotation_gate_mode or "clamp",
        }

    ref_rows = load_pose_csv(args.reference_csv) if args.reference_csv else {}
    model = load_stl_as_pointcloud(args.stl, icp_cfg["n_model_points"])
    model = downsample_cloud(model, icp_cfg["voxel_size_m"])
    stl_extent = get_projectable_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()
    if depth_scale is None:
        raise RuntimeError("No depth scale found in bag.")

    out_frames = args.out_dir / "frames"
    color_depth_dir = out_frames / "color_depth"
    pose_gate_dir = out_frames / "pose_gate"
    masks_dir = out_frames / "masks"
    bbox_dir = out_frames / "bbox"
    clusters_dir = args.out_dir / "clusters"
    for directory in [color_depth_dir, pose_gate_dir, masks_dir, bbox_dir, clusters_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    out_csv = args.out_dir / "poses_cluster_assisted.csv"
    out_video = args.out_dir / "cluster_assisted_overlay.mp4"
    writer = None
    previous_pose = None
    previous_mask = None
    rows = []
    processed = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.start_frame + args.max_frames):
        if frame.frame_id < args.start_frame:
            continue

        raw_points, raw_colors, raw_pixels, roi_box = backproject_candidates(
            frame,
            intrinsics,
            depth_scale,
            roi,
            depth_cfg,
            color_cfg,
        )
        gate_margin = float(cluster_cfg.get("pose_gate_margin_m", icp_cfg.get("pose_gate_margin_m", 0.012)))
        min_gate_points = int(cluster_cfg.get("pose_gate_min_points", cluster_cfg["min_cluster_points"]))
        points, colors, pixels, gate_keep, gate_used = filter_by_previous_pose_gate(
            raw_points,
            raw_colors,
            raw_pixels,
            previous_pose,
            stl_extent,
            gate_margin,
            min_gate_points,
        )
        gate_mask = np.zeros(frame.depth_z16.shape, dtype=np.uint8)
        if gate_keep is not None:
            gate_mask[raw_pixels[gate_keep, 1], raw_pixels[gate_keep, 0]] = 255
        elif previous_pose is None:
            gate_mask[raw_pixels[:, 1], raw_pixels[:, 0]] = 255

        selected, labels, candidates = select_cluster(
            points,
            colors,
            pixels,
            cluster_cfg,
            stl_extent,
            previous_pose,
            previous_mask,
        )
        ref_t = None
        ref_row = ref_rows.get(frame.frame_id)
        if ref_row is not None and ref_row["detected"] == "1":
            ref_t = row_to_transform(ref_row)

        if selected is None:
            row = {
                "frame_id": frame.frame_id,
                "timestamp": f"{frame.timestamp_s:.6f}",
                "detected": 0,
                "selected_label": "",
                "selected_points": 0,
                "raw_candidate_points": len(raw_points),
                "pose_gate_points": int(np.count_nonzero(gate_keep)) if gate_keep is not None else len(raw_points),
                "pose_gate_used": int(gate_used),
                "init_fitness": "",
                "init_rmse": "",
                "icp_fitness_raw": "",
                "icp_rmse_raw": "",
                "icp_translation_step_m": "",
                "icp_rotation_step_deg": "",
                "rotation_bounded": 0,
                "icp_accepted": 0,
                "icp_rejection_reason": "no_cluster",
                "fitness": "0.000000000",
                "rmse": "0.000000000",
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
            rows.append(row)
            processed += 1
            continue

        selected_idx = selected["idx"]
        selected_points = points[selected_idx]
        selected_colors = colors[selected_idx]
        observed = o3d.geometry.PointCloud()
        observed.points = o3d.utility.Vector3dVector(selected_points)
        observed.colors = o3d.utility.Vector3dVector(selected_colors)
        observed = downsample_cloud(observed, icp_cfg["voxel_size_m"])

        orientation_mode = cluster_cfg.get("orientation_mode", "pca_each_frame")
        use_previous_orientation = orientation_mode == "previous_after_first" and previous_pose is not None
        init_t = pose_from_cluster_translation_and_previous_orientation(
            selected["pose"],
            selected_points,
            stl_extent,
            previous_pose,
            use_previous_orientation,
        )
        init_fitness, init_rmse = evaluate_alignment(
            model,
            observed,
            init_t,
            icp_cfg["max_correspondence_m"],
        )
        icp_t, fitness, rmse = run_point_to_point_icp(
            model,
            observed,
            init_t,
            icp_cfg["max_correspondence_m"],
            icp_cfg["max_iterations"],
        )
        if args.lock_rotation:
            locked = np.eye(4, dtype=np.float64)
            locked[:3, :3] = init_t[:3, :3]
            locked[:3, 3] = icp_t[:3, 3]
            icp_t = locked
            eval_result = o3d.pipelines.registration.evaluate_registration(
                model,
                observed,
                icp_cfg["max_correspondence_m"],
                icp_t,
            )
            fitness = float(eval_result.fitness)
            rmse = float(eval_result.inlier_rmse)

        raw_icp_t = icp_t.copy()
        raw_icp_fitness = fitness
        raw_icp_rmse = rmse
        bounded_icp_t, icp_rotation_step, rotation_bounded = bound_rotation_update(
            init_t,
            raw_icp_t,
            icp_cfg.get("rotation_gate", {}),
        )
        final_t, icp_accepted, rejection_reason = choose_icp_pose(
            init_t,
            bounded_icp_t,
            init_fitness,
            init_rmse,
            raw_icp_fitness,
            raw_icp_rmse,
            icp_cfg,
        )
        fitness, rmse = evaluate_alignment(
            model,
            observed,
            final_t,
            icp_cfg["max_correspondence_m"],
        )
        icp_translation_step = float(np.linalg.norm(raw_icp_t[:3, 3] - init_t[:3, 3]))

        previous_pose = final_t
        quat = matrix_to_quaternion_xyzw(final_t)
        ref_translation_error = ""
        ref_rotation_error = ""
        if ref_t is not None:
            ref_translation_error = f"{np.linalg.norm(final_t[:3, 3] - ref_t[:3, 3]):.9f}"
            ref_rotation_error = f"{rotation_error_deg(ref_t, final_t):.6f}"

        row = {
            "frame_id": frame.frame_id,
            "timestamp": f"{frame.timestamp_s:.6f}",
            "detected": 1,
            "selected_label": selected["label"],
            "selected_points": len(observed.points),
            "raw_candidate_points": len(raw_points),
            "pose_gate_points": int(np.count_nonzero(gate_keep)) if gate_keep is not None else len(raw_points),
            "pose_gate_used": int(gate_used),
            "orientation_source": "previous" if use_previous_orientation else "cluster_pca",
            "init_fitness": f"{init_fitness:.9f}",
            "init_rmse": f"{init_rmse:.9f}",
            "icp_fitness_raw": f"{raw_icp_fitness:.9f}",
            "icp_rmse_raw": f"{raw_icp_rmse:.9f}",
            "icp_translation_step_m": f"{icp_translation_step:.9f}",
            "icp_rotation_step_deg": f"{icp_rotation_step:.6f}",
            "rotation_bounded": int(rotation_bounded),
            "icp_accepted": int(icp_accepted),
            "icp_rejection_reason": rejection_reason,
            "fitness": f"{fitness:.9f}",
            "rmse": f"{rmse:.9f}",
            "tx": f"{final_t[0, 3]:.9f}",
            "ty": f"{final_t[1, 3]:.9f}",
            "tz": f"{final_t[2, 3]:.9f}",
            "qx": f"{quat[0]:.9f}",
            "qy": f"{quat[1]:.9f}",
            "qz": f"{quat[2]:.9f}",
            "qw": f"{quat[3]:.9f}",
            "ref_translation_error_m": ref_translation_error,
            "ref_rotation_error_deg": ref_rotation_error,
        }
        rows.append(row)

        save_cluster_summary(clusters_dir / f"frame_{frame.frame_id:06d}_clusters.csv", candidates, selected["label"])
        mask = np.zeros(frame.depth_z16.shape, dtype=np.uint8)
        mask[pixels[selected_idx, 1], pixels[selected_idx, 0]] = 255
        dilate_px = int(cluster_cfg.get("previous_mask_dilate_px", 0))
        if dilate_px > 0:
            kernel_size = max(3, 2 * (dilate_px // 2) + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            previous_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            previous_mask = mask

        color_depth, mask_overlay, bbox_overlay = draw_frame_outputs(
            frame,
            mask,
            init_t,
            final_t,
            ref_t,
            intrinsics,
            stl_extent,
            roi_box,
            selected["label"],
        )
        gate_overlay = draw_gate_overlay(frame, gate_mask, roi_box, gate_used)
        cv2.imwrite(str(color_depth_dir / f"frame_{frame.frame_id:06d}_color_depth.png"), color_depth)
        cv2.imwrite(str(pose_gate_dir / f"frame_{frame.frame_id:06d}_pose_gate.png"), gate_overlay)
        cv2.imwrite(str(masks_dir / f"frame_{frame.frame_id:06d}_mask_overlay.png"), mask_overlay)
        cv2.imwrite(str(masks_dir / f"frame_{frame.frame_id:06d}_mask.png"), mask)
        cv2.imwrite(str(bbox_dir / f"frame_{frame.frame_id:06d}_bbox_overlay.png"), bbox_overlay)

        if writer is None:
            height, width = bbox_overlay.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (width, height))
        writer.write(bbox_overlay)
        processed += 1

    if writer is not None:
        writer.release()

    fieldnames = [
        "frame_id",
        "timestamp",
        "detected",
        "selected_label",
        "selected_points",
        "raw_candidate_points",
        "pose_gate_points",
        "pose_gate_used",
        "orientation_source",
        "init_fitness",
        "init_rmse",
        "icp_fitness_raw",
        "icp_rmse_raw",
        "icp_translation_step_m",
        "icp_rotation_step_deg",
        "rotation_bounded",
        "icp_accepted",
        "icp_rejection_reason",
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

    print(f"processed {processed} frames")
    print(f"saved poses to {out_csv}")
    print(f"saved overlay video to {out_video}")
    print(f"saved per-frame diagnostics under {out_frames}")


if __name__ == "__main__":
    main()
