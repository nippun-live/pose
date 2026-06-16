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
from plug_pose.visualization import draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def transform_to_row(frame_id: int, transform: np.ndarray, detected: int = 1) -> dict[str, str | int]:
    quat = matrix_to_quaternion_xyzw(transform)
    return {
        "frame_id": frame_id,
        "detected": detected,
        "tx": f"{transform[0, 3]:.9f}",
        "ty": f"{transform[1, 3]:.9f}",
        "tz": f"{transform[2, 3]:.9f}",
        "qx": f"{quat[0]:.9f}",
        "qy": f"{quat[1]:.9f}",
        "qz": f"{quat[2]:.9f}",
        "qw": f"{quat[3]:.9f}",
    }


def write_pose_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def estimate_cluster_pca_pose(observed: o3d.geometry.PointCloud, stl_extent: np.ndarray) -> np.ndarray:
    points = np.asarray(observed.points)
    if len(points) < 20:
        raise RuntimeError("Need at least 20 observed points for PCA initialization.")

    surface_center = points.mean(axis=0)
    centered = points - surface_center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    x_axis = eigvecs[:, 0]
    z_axis = eigvecs[:, 2]

    # Visible surface normal should point toward the camera; camera center is the origin.
    to_camera = -surface_center
    to_camera /= np.linalg.norm(to_camera)
    if np.dot(z_axis, to_camera) < 0:
        z_axis = -z_axis

    # For frame 127, choose +X along the plug from marker end toward cable end
    # in the image, which is mostly toward negative camera X.
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    if x_axis[0] > 0:
        x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    # The cluster is mostly the visible face. Shift from that face into the plug.
    transform[:3, 3] = surface_center - z_axis * (float(stl_extent[2]) / 2.0)
    return transform


def draw_overlay(
    bag: Path,
    frame_id: int,
    mask_path: Path,
    stl: Path,
    init_t: np.ndarray,
    icp_t: np.ndarray,
    out_image: Path,
    reference_t: np.ndarray | None,
) -> None:
    pipeline, profile = start_playback(bag)
    try:
        intrinsics = get_color_intrinsics(profile)
    finally:
        pipeline.stop()

    frame = None
    for candidate in iter_aligned_frames(bag, max_frames=frame_id + 1):
        frame = candidate
    if frame is None or frame.frame_id != frame_id:
        raise RuntimeError(f"Could not read frame {frame_id}")

    image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        blue = np.zeros_like(image_bgr)
        blue[:, :, 0] = 255
        image_bgr = np.where(mask[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.45, blue, 0.55, 0), image_bgr)

    extent = get_projectable_stl_bbox_extent(stl)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    init_rvec, init_tvec = matrix_to_rvec_tvec(init_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, init_rvec, init_tvec, extent, color=(0, 165, 255), thickness=2)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    icp_rvec, icp_tvec = matrix_to_rvec_tvec(icp_t)
    image_bgr = draw_projected_bbox(image_rgb, intrinsics, icp_rvec, icp_tvec, extent, color=(0, 255, 0), thickness=2)

    if reference_t is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ref_rvec, ref_tvec = matrix_to_rvec_tvec(reference_t)
        image_bgr = draw_projected_bbox(image_rgb, intrinsics, ref_rvec, ref_tvec, extent, color=(255, 0, 255), thickness=2)
        label = "blue:depth  orange:cluster init  green:ICP  magenta:ArUco ref"
    else:
        label = "blue:depth  orange:cluster init  green:ICP"

    cv2.putText(image_bgr, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), image_bgr)


def write_debug_clouds(debug_dir: Path, frame_id: int, observed, model, init_t, icp_t) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    observed_out = transformed_copy(observed, np.eye(4)).paint_uniform_color([0.1, 0.7, 1.0])
    init_out = transformed_copy(model, init_t).paint_uniform_color([1.0, 0.55, 0.1])
    icp_out = transformed_copy(model, icp_t).paint_uniform_color([0.1, 1.0, 0.25])
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_observed_selected.ply"), observed_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_cluster_init.ply"), init_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_model_icp.ply"), icp_out)
    o3d.io.write_point_cloud(str(debug_dir / f"frame_{frame_id:06d}_combined.ply"), observed_out + init_out + icp_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--observed_ply", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--reference_csv", type=Path)
    parser.add_argument("--lock_rotation", action="store_true")
    parser.add_argument("--out_init_csv", type=Path, required=True)
    parser.add_argument("--out_icp_csv", type=Path, required=True)
    parser.add_argument("--out_image", type=Path, required=True)
    parser.add_argument("--debug_dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    icp_cfg = config["markerless"]["icp"]
    stl_extent = get_projectable_stl_bbox_extent(args.stl)

    observed = o3d.io.read_point_cloud(str(args.observed_ply))
    observed = downsample_cloud(observed, icp_cfg["voxel_size_m"])
    model = load_stl_as_pointcloud(args.stl, icp_cfg["n_model_points"])
    model = downsample_cloud(model, icp_cfg["voxel_size_m"])

    init_t = estimate_cluster_pca_pose(observed, stl_extent)
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

    reference_t = None
    ref_translation_error = ""
    ref_rotation_error = ""
    if args.reference_csv:
        ref_rows = load_pose_csv(args.reference_csv)
        if args.frame_id in ref_rows and ref_rows[args.frame_id]["detected"] == "1":
            reference_t = row_to_transform(ref_rows[args.frame_id])
            ref_translation_error = f"{np.linalg.norm(icp_t[:3, 3] - reference_t[:3, 3]):.9f}"
            ref_rotation_error = f"{rotation_error_deg(reference_t, icp_t):.6f}"

    init_row = transform_to_row(args.frame_id, init_t)
    icp_row = transform_to_row(args.frame_id, icp_t)
    icp_row.update(
        {
            "fitness": f"{fitness:.9f}",
            "rmse": f"{rmse:.9f}",
            "init_translation_delta_m": f"{np.linalg.norm(icp_t[:3, 3] - init_t[:3, 3]):.9f}",
            "init_rotation_delta_deg": f"{rotation_error_deg(init_t, icp_t):.6f}",
            "ref_translation_error_m": ref_translation_error,
            "ref_rotation_error_deg": ref_rotation_error,
        }
    )

    write_pose_csv(args.out_init_csv, init_row)
    write_pose_csv(args.out_icp_csv, icp_row)
    write_debug_clouds(args.debug_dir, args.frame_id, observed, model, init_t, icp_t)
    draw_overlay(args.bag, args.frame_id, args.mask, args.stl, init_t, icp_t, args.out_image, reference_t)

    print(f"observed selected points after downsample: {len(observed.points)}")
    print(f"model points after downsample: {len(model.points)}")
    print(f"fitness: {fitness:.9f}")
    print(f"rmse: {rmse:.9f}")
    print(f"init translation delta m: {icp_row['init_translation_delta_m']}")
    print(f"init rotation delta deg: {icp_row['init_rotation_delta_deg']}")
    if reference_t is not None:
        print(f"reference translation error m: {ref_translation_error}")
        print(f"reference rotation error deg: {ref_rotation_error}")
    print(f"saved cluster init CSV to {args.out_init_csv}")
    print(f"saved ICP CSV to {args.out_icp_csv}")
    print(f"saved overlay image to {args.out_image}")
    print(f"saved debug clouds to {args.debug_dir}")


if __name__ == "__main__":
    main()
