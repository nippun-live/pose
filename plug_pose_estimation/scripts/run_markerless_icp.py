from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.icp_pose import run_point_to_point_icp, transformed_copy  # noqa: E402
from plug_pose.pointcloud import backproject_depth_roi, downsample_cloud, filter_by_pose_bbox  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.stl_utils import load_stl_as_pointcloud  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_pose_axes, draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def write_debug_clouds(
    debug_dir: Path,
    frame_id: int,
    observed: o3d.geometry.PointCloud,
    model: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    icp_transform: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    observed_out = observed.paint_uniform_color([0.1, 0.7, 1.0])
    init_out = transformed_copy(model, init_transform).paint_uniform_color([1.0, 0.55, 0.1])
    icp_out = transformed_copy(model, icp_transform).paint_uniform_color([0.1, 1.0, 0.25])
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_observed.ply"), observed_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_init.ply"), init_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_icp.ply"), icp_out)
    combined = observed_out + init_out + icp_out
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_combined.ply"), combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--roi_key", default="roi_tagged")
    parser.add_argument("--init_csv", type=Path, help="Optional per-frame initialization poses.")
    parser.add_argument(
        "--init_first_only",
        action="store_true",
        help="Use init_csv only at start_frame, then track from the previous ICP pose.",
    )
    parser.add_argument("--reference_csv", type=Path, help="Optional reference poses for comparison.")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_video", type=Path)
    parser.add_argument("--debug_dir", type=Path)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1)
    parser.add_argument(
        "--lock_rotation",
        choices=["config", "true", "false"],
        default="config",
        help="Override config markerless.icp.lock_rotation.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    markerless = config["markerless"]
    roi = markerless[args.roi_key]
    depth_config = markerless["depth"]
    icp_config = markerless["icp"]
    if args.lock_rotation == "true":
        lock_rotation = True
    elif args.lock_rotation == "false":
        lock_rotation = False
    else:
        lock_rotation = bool(icp_config.get("lock_rotation", False))

    init_poses = load_pose_csv(args.init_csv) if args.init_csv else {}
    reference_poses = load_pose_csv(args.reference_csv) if args.reference_csv else {}

    model_pcd = load_stl_as_pointcloud(args.stl, icp_config["n_model_points"])
    model_pcd = downsample_cloud(model_pcd, icp_config["voxel_size_m"])
    bbox_extent = get_projectable_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    if depth_scale is None:
        raise RuntimeError("No depth scale found in bag.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    previous_pose = None
    rows = []
    processed = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.start_frame + args.max_frames):
        if frame.frame_id < args.start_frame:
            continue

        observed = backproject_depth_roi(
            frame.depth_z16,
            frame.color_rgb,
            intrinsics,
            depth_scale,
            roi,
            depth_config["min_m"],
            depth_config["max_m"],
        )
        observed = downsample_cloud(observed, icp_config["voxel_size_m"])

        use_csv_init = (
            frame.frame_id in init_poses
            and init_poses[frame.frame_id]["detected"] == "1"
            and (not args.init_first_only or frame.frame_id == args.start_frame or previous_pose is None)
        )
        if use_csv_init:
            init_transform = row_to_transform(init_poses[frame.frame_id])
        elif previous_pose is not None:
            init_transform = previous_pose
        else:
            init_transform = np.eye(4, dtype=np.float64)
            if len(observed.points) >= 20:
                init_transform[:3, 3] = np.asarray(observed.get_center())

        if "pose_gate_margin_m" in icp_config:
            observed = filter_by_pose_bbox(
                observed,
                init_transform,
                bbox_extent,
                icp_config["pose_gate_margin_m"],
            )

        detected = len(observed.points) >= 20
        fitness = 0.0
        rmse = 0.0
        icp_transform = init_transform
        if detected:
            icp_transform, fitness, rmse = run_point_to_point_icp(
                model_pcd,
                observed,
                init_transform,
                icp_config["max_correspondence_m"],
                icp_config["max_iterations"],
            )
            if lock_rotation:
                locked_transform = np.eye(4, dtype=np.float64)
                locked_transform[:3, :3] = init_transform[:3, :3]
                locked_transform[:3, 3] = icp_transform[:3, 3]
                icp_transform = locked_transform
                evaluation = o3d.pipelines.registration.evaluate_registration(
                    model_pcd,
                    observed,
                    icp_config["max_correspondence_m"],
                    icp_transform,
                )
                fitness = float(evaluation.fitness)
                rmse = float(evaluation.inlier_rmse)
            previous_pose = icp_transform

        quat = matrix_to_quaternion_xyzw(icp_transform)
        row = {
            "frame_id": frame.frame_id,
            "timestamp": f"{frame.timestamp_s:.6f}",
            "detected": int(detected),
            "tx": f"{icp_transform[0, 3]:.9f}",
            "ty": f"{icp_transform[1, 3]:.9f}",
            "tz": f"{icp_transform[2, 3]:.9f}",
            "qx": f"{quat[0]:.9f}",
            "qy": f"{quat[1]:.9f}",
            "qz": f"{quat[2]:.9f}",
            "qw": f"{quat[3]:.9f}",
            "fitness": f"{fitness:.9f}",
            "rmse": f"{rmse:.9f}",
            "observed_points": len(observed.points),
        }

        ref_row = reference_poses.get(frame.frame_id)
        if ref_row is not None and ref_row["detected"] == "1":
            ref_transform = row_to_transform(ref_row)
            translation_error = np.linalg.norm(icp_transform[:3, 3] - ref_transform[:3, 3])
            rotation_delta = ref_transform[:3, :3].T @ icp_transform[:3, :3]
            angle = np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
            row["ref_translation_error_m"] = f"{translation_error:.9f}"
            row["ref_rotation_error_deg"] = f"{np.degrees(angle):.6f}"
        else:
            row["ref_translation_error_m"] = ""
            row["ref_rotation_error_deg"] = ""

        rows.append(row)

        output_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        if detected:
            rvec, tvec = matrix_to_rvec_tvec(icp_transform)
            bbox_bgr = draw_projected_bbox(frame.color_rgb, intrinsics, rvec, tvec, bbox_extent)
            bbox_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)
            output_bgr = draw_pose_axes(bbox_rgb, intrinsics, rvec, tvec, 0.02)

        if args.out_video:
            if writer is None:
                height, width = output_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.out_video), fourcc, float(fps), (width, height))
            writer.write(output_bgr)

        if args.debug_dir and processed == 0:
            write_debug_clouds(args.debug_dir, frame.frame_id, observed, model_pcd, init_transform, icp_transform)

        processed += 1

    if writer is not None:
        writer.release()

    fieldnames = [
        "frame_id",
        "timestamp",
        "detected",
        "tx",
        "ty",
        "tz",
        "qx",
        "qy",
        "qz",
        "qw",
        "fitness",
        "rmse",
        "observed_points",
        "ref_translation_error_m",
        "ref_rotation_error_deg",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"processed {processed} frames")
    print(f"saved poses to {args.out_csv}")
    if args.out_video:
        print(f"saved overlay video to {args.out_video}")
    if args.debug_dir:
        print(f"saved debug point clouds to {args.debug_dir}")


if __name__ == "__main__":
    main()
