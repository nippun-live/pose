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

from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.icp_pose import run_point_to_point_icp, transformed_copy  # noqa: E402
from plug_pose.pointcloud import downsample_cloud  # noqa: E402
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


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def write_debug_clouds(debug_dir: Path, frame_id: int, observed, model, init_t, icp_t) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    observed_out = transformed_copy(observed, np.eye(4)).paint_uniform_color([0.1, 0.7, 1.0])
    init_out = transformed_copy(model, init_t).paint_uniform_color([1.0, 0.55, 0.1])
    icp_out = transformed_copy(model, icp_t).paint_uniform_color([0.1, 1.0, 0.25])
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_observed_selected.ply"), observed_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_init.ply"), init_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_icp.ply"), icp_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_combined.ply"), observed_out + init_out + icp_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--observed_ply", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--init_csv", type=Path, required=True)
    parser.add_argument("--reference_csv", type=Path)
    parser.add_argument("--lock_rotation", action="store_true")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_image", type=Path, required=True)
    parser.add_argument("--debug_dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    icp_cfg = config["markerless"]["icp"]

    observed = o3d.io.read_point_cloud(str(args.observed_ply))
    if observed.is_empty():
        raise RuntimeError(f"Observed cloud is empty: {args.observed_ply}")
    observed = downsample_cloud(observed, icp_cfg["voxel_size_m"])

    model = load_stl_as_pointcloud(args.stl, icp_cfg["n_model_points"])
    model = downsample_cloud(model, icp_cfg["voxel_size_m"])

    init_rows = load_pose_csv(args.init_csv)
    if args.frame_id not in init_rows or init_rows[args.frame_id]["detected"] != "1":
        raise RuntimeError(f"No detected init pose for frame {args.frame_id}")
    init_t = row_to_transform(init_rows[args.frame_id])

    ref_t = None
    if args.reference_csv:
        ref_rows = load_pose_csv(args.reference_csv)
        if args.frame_id in ref_rows and ref_rows[args.frame_id]["detected"] == "1":
            ref_t = row_to_transform(ref_rows[args.frame_id])

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

    quat = matrix_to_quaternion_xyzw(icp_t)
    row = {
        "frame_id": args.frame_id,
        "detected": 1,
        "tx": f"{icp_t[0, 3]:.9f}",
        "ty": f"{icp_t[1, 3]:.9f}",
        "tz": f"{icp_t[2, 3]:.9f}",
        "qx": f"{quat[0]:.9f}",
        "qy": f"{quat[1]:.9f}",
        "qz": f"{quat[2]:.9f}",
        "qw": f"{quat[3]:.9f}",
        "fitness": f"{fitness:.9f}",
        "rmse": f"{rmse:.9f}",
        "init_translation_delta_m": f"{np.linalg.norm(icp_t[:3, 3] - init_t[:3, 3]):.9f}",
        "init_rotation_delta_deg": f"{rotation_error_deg(init_t, icp_t):.6f}",
        "ref_translation_error_m": "",
        "ref_rotation_error_deg": "",
    }
    if ref_t is not None:
        row["ref_translation_error_m"] = f"{np.linalg.norm(icp_t[:3, 3] - ref_t[:3, 3]):.9f}"
        row["ref_rotation_error_deg"] = f"{rotation_error_deg(ref_t, icp_t):.6f}"

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    write_debug_clouds(args.debug_dir, args.frame_id, observed, model, init_t, icp_t)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
    finally:
        pipeline.stop()

    frame = None
    for candidate in iter_aligned_frames(args.bag, max_frames=args.frame_id + 1):
        frame = candidate
    if frame is None or frame.frame_id != args.frame_id:
        raise RuntimeError(f"Could not read frame {args.frame_id}")

    bbox_extent = get_projectable_stl_bbox_extent(args.stl)
    color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    init_rvec, init_tvec = matrix_to_rvec_tvec(init_t)
    init_box = draw_projected_bbox(frame.color_rgb, intrinsics, init_rvec, init_tvec, bbox_extent, color=(0, 165, 255))
    init_box_rgb = cv2.cvtColor(init_box, cv2.COLOR_BGR2RGB)
    icp_rvec, icp_tvec = matrix_to_rvec_tvec(icp_t)
    both = draw_projected_bbox(init_box_rgb, intrinsics, icp_rvec, icp_tvec, bbox_extent, color=(0, 255, 0))
    both_rgb = cv2.cvtColor(both, cv2.COLOR_BGR2RGB)
    both = draw_pose_axes(both_rgb, intrinsics, icp_rvec, icp_tvec, 0.02)
    cv2.putText(both, "orange:init  green:ICP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_image), both)

    print(f"observed selected points after downsample: {len(observed.points)}")
    print(f"model points after downsample: {len(model.points)}")
    print(f"fitness: {fitness:.9f}")
    print(f"rmse: {rmse:.9f}")
    print(f"init translation delta m: {row['init_translation_delta_m']}")
    print(f"init rotation delta deg: {row['init_rotation_delta_deg']}")
    if ref_t is not None:
        print(f"reference translation error m: {row['ref_translation_error_m']}")
        print(f"reference rotation error deg: {row['ref_rotation_error_deg']}")
    print(f"saved CSV to {args.out_csv}")
    print(f"saved overlay image to {args.out_image}")
    print(f"saved debug clouds to {args.debug_dir}")


if __name__ == "__main__":
    main()
