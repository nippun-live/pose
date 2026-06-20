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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.icp_pose import run_icp_variant  # noqa: E402
from plug_pose.pointcloud import downsample_cloud  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.stl_utils import load_stl_as_pointcloud  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
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
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    output: dict[int, Path] = {}
    for row in rows:
        frame_id = int(row["bag_frame_id"])
        mask_path = Path(row["mask"])
        if not mask_path.is_absolute() and not mask_path.exists():
            mask_path = path.parent / "masks" / f"mask_{frame_id:06d}.png"
        output[frame_id] = mask_path
    return output


def dominant_depth_mask(depth_m: np.ndarray, mask: np.ndarray, min_m: float, max_m: float, band_m: float) -> np.ndarray:
    candidate = (mask > 0) & (depth_m > min_m) & (depth_m < max_m)
    values = depth_m[candidate]
    if len(values) < 50:
        return candidate
    hist, bins = np.histogram(values, bins=80, range=(min_m, max_m))
    peak_idx = int(np.argmax(hist))
    peak = float((bins[peak_idx] + bins[peak_idx + 1]) / 2.0)
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
        float(depth_cfg.get("max_m", 0.45)),
        band_m,
    )
    vs, us = np.nonzero(keep)
    z = depth_m[vs, us]
    x = (us - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vs - intrinsics["ppy"]) * z / intrinsics["fy"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    pcd.colors = o3d.utility.Vector3dVector(frame.color_rgb[vs, us].astype(np.float64) / 255.0)
    return pcd, keep.astype(np.uint8) * 255


def estimate_first_pose(points: np.ndarray, stl_extent: np.ndarray) -> np.ndarray:
    surface_center = np.median(points, axis=0)
    centered = points - surface_center
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

    x_axis = eigvecs[:, 0]
    z_axis = eigvecs[:, 2]
    to_camera = -surface_center
    to_camera /= np.linalg.norm(to_camera)
    if np.dot(z_axis, to_camera) < 0:
        z_axis = -z_axis
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    if x_axis[0] > 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = surface_center - z_axis * (float(stl_extent[2]) / 2.0)
    return pose


def initialize_pose(points: np.ndarray, stl_extent: np.ndarray, previous_pose: np.ndarray | None) -> np.ndarray:
    if previous_pose is None:
        return estimate_first_pose(points, stl_extent)

    local = (previous_pose[:3, :3].T @ (points - previous_pose[:3, 3]).T).T
    center_local = np.median(local, axis=0)
    center_local[2] = np.median(local[:, 2]) - float(stl_extent[2]) / 2.0

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = previous_pose[:3, :3]
    pose[:3, 3] = previous_pose[:3, 3] + previous_pose[:3, :3] @ center_local
    return pose


def draw_overlay(frame, depth_mask, init_t, final_t, ref_t, intrinsics, stl_extent):
    image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    blue = np.zeros_like(image_bgr)
    blue[:, :, 0] = 255
    image_bgr = np.where(depth_mask[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.45, blue, 0.55, 0), image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rvec, tvec = matrix_to_rvec_tvec(init_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, rvec, tvec, stl_extent, color=(0, 165, 255), thickness=2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rvec, tvec = matrix_to_rvec_tvec(final_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, rvec, tvec, stl_extent, color=(0, 255, 0), thickness=2)

    text = "blue:mask+depth orange:init green:ICP"
    if ref_t is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rvec, tvec = matrix_to_rvec_tvec(ref_t)
        image_bgr = draw_projected_bbox(image_rgb, intrinsics, rvec, tvec, stl_extent, color=(255, 0, 255), thickness=2)
        text += " magenta:ArUco ref"

    cv2.putText(image_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image_bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final SAM2-mask + RGB-D + STL multiscale P2P ICP tracker.")
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
    parser.add_argument("--save_debug_clouds", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    depth_cfg = config["markerless"]["depth"]
    icp_cfg = config["markerless"]["icp"]
    mask_paths = load_mask_map(args.mask_map)
    refs = load_pose_csv(args.reference_csv)

    model_full = load_stl_as_pointcloud(args.stl, int(icp_cfg["n_model_points"]))
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
    output_dirs = [overlay_dir, depth_mask_dir]
    if args.save_debug_clouds:
        output_dirs.append(cloud_dir)
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    writer = None
    previous_pose = None
    end_frame = None if args.max_frames is None else args.start_frame + args.max_frames

    for frame in iter_aligned_frames(args.bag, max_frames=end_frame):
        if frame.frame_id < args.start_frame:
            continue
        mask_path = mask_paths.get(frame.frame_id)
        if mask_path is None:
            continue

        observed_raw, depth_mask = backproject_mask(frame, mask_path, intrinsics, depth_scale, depth_cfg, args.depth_band_m)
        observed = downsample_cloud(observed_raw, float(icp_cfg["voxel_size_m"]))
        ref_t = None
        ref_row = refs.get(frame.frame_id)
        if ref_row is not None and ref_row.get("detected") == "1" and ref_row.get("tx"):
            ref_t = row_to_transform(ref_row)

        row = {
            "frame_id": frame.frame_id,
            "timestamp": f"{frame.timestamp_s:.6f}",
            "detected": 0,
            "icp_method": "multiscale_point_to_point",
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

        if len(observed.points) >= int(args.min_observed_points):
            init_t = initialize_pose(np.asarray(observed.points), stl_extent, previous_pose)
            icp_t, fitness, rmse = run_icp_variant(
                model_full,
                observed_raw,
                init_t,
                "multiscale_point_to_point",
                float(icp_cfg["max_correspondence_m"]),
                int(icp_cfg["max_iterations"]),
            )
            previous_pose = icp_t
            quat = matrix_to_quaternion_xyzw(icp_t)
            row.update(
                {
                    "detected": 1,
                    "model_points": len(model_full.points),
                    "fitness": f"{fitness:.9f}",
                    "rmse": f"{rmse:.9f}",
                    "tx": f"{icp_t[0, 3]:.9f}",
                    "ty": f"{icp_t[1, 3]:.9f}",
                    "tz": f"{icp_t[2, 3]:.9f}",
                    "qx": f"{quat[0]:.9f}",
                    "qy": f"{quat[1]:.9f}",
                    "qz": f"{quat[2]:.9f}",
                    "qw": f"{quat[3]:.9f}",
                }
            )
            if ref_t is not None:
                row["ref_translation_error_m"] = f"{np.linalg.norm(icp_t[:3, 3] - ref_t[:3, 3]):.9f}"
                row["ref_rotation_error_deg"] = f"{rotation_error_deg(ref_t, icp_t):.6f}"
            overlay = draw_overlay(frame, depth_mask, init_t, icp_t, ref_t, intrinsics, stl_extent)
        else:
            overlay = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(overlay_dir / f"frame_{frame.frame_id:06d}_overlay.png"), overlay)
        cv2.imwrite(str(depth_mask_dir / f"frame_{frame.frame_id:06d}_depth_mask.png"), depth_mask)
        if args.save_debug_clouds and frame.frame_id % 20 == 0:
            o3d.io.write_point_cloud(str(cloud_dir / f"frame_{frame.frame_id:06d}_observed.ply"), observed)

        if writer is None:
            h, w = overlay.shape[:2]
            writer = cv2.VideoWriter(str(args.out_dir / "markerless_pose_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        writer.write(overlay)
        rows.append(row)

    if writer is not None:
        writer.release()

    out_csv = args.out_dir / "poses_markerless.csv"
    if not rows:
        raise RuntimeError("No frames were processed. Check --mask_map, --start_frame, and --max_frames.")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"saved {len(rows)} pose rows to {out_csv}")
    print(f"saved overlay video to {args.out_dir / 'markerless_pose_overlay.mp4'}")


if __name__ == "__main__":
    main()
