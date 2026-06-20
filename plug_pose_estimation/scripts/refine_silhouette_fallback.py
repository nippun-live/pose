from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, intrinsics_to_camera_matrix, intrinsics_to_dist_coeffs, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_projected_bbox  # noqa: E402


def parse_ranges(values: list[str]) -> set[int]:
    frames: set[int] = set()
    for value in values:
        if "-" in value:
            start, end = [int(part) for part in value.split("-", 1)]
            frames.update(range(start, end + 1))
        else:
            frames.add(int(value))
    return frames


def load_pose_rows(path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {int(row["frame_id"]): row for row in rows}


def load_mask_map(path: Path) -> dict[int, Path]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    masks = {}
    for row in rows:
        frame_id = int(row["bag_frame_id"])
        mask_path = Path(row["mask"])
        if not mask_path.is_absolute() and not mask_path.exists():
            mask_path = path.parent / "masks" / f"mask_{frame_id:06d}.png"
        masks[frame_id] = mask_path
    return masks


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def bbox_corners(extent_xyz_m: np.ndarray) -> np.ndarray:
    ex, ey, ez = np.asarray(extent_xyz_m, dtype=np.float32) / 2.0
    return np.array(
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


def delta_pose(base_pose: np.ndarray, params: np.ndarray) -> np.ndarray:
    tx, ty, tz, rx, ry, rz = params
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    delta[:3, 3] = [tx, ty, tz]
    return delta @ base_pose


def projected_bbox_mask(
    transform: np.ndarray,
    corners: np.ndarray,
    intrinsics: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    rvec, tvec = matrix_to_rvec_tvec(transform)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ corners.astype(np.float64).T).T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    if np.any(camera_points[:, 2] <= 0.01):
        return np.zeros(shape, dtype=np.uint8)

    points, _ = cv2.projectPoints(
        corners,
        rvec,
        tvec,
        intrinsics_to_camera_matrix(intrinsics),
        intrinsics_to_dist_coeffs(intrinsics),
    )
    points = points.reshape(-1, 2)
    if not np.isfinite(points).all():
        return np.zeros(shape, dtype=np.uint8)

    height, width = shape
    if np.any(np.abs(points) > max(height, width) * 4):
        return np.zeros(shape, dtype=np.uint8)

    hull = cv2.convexHull(np.round(points).astype(np.int32))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def optimize_to_mask(
    base_pose: np.ndarray,
    sam_mask: np.ndarray,
    corners: np.ndarray,
    intrinsics: dict,
    max_translation_m: float,
    max_z_m: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, float, float]:
    target = (sam_mask > 0).astype(np.uint8) * 255
    base_projection = projected_bbox_mask(base_pose, corners, intrinsics, target.shape)
    base_iou = mask_iou(base_projection, target)
    max_rot_rad = np.deg2rad(max_rotation_deg)
    bounds = [
        (-max_translation_m, max_translation_m),
        (-max_translation_m, max_translation_m),
        (-max_z_m, max_z_m),
        (-max_rot_rad, max_rot_rad),
        (-max_rot_rad, max_rot_rad),
        (-max_rot_rad, max_rot_rad),
    ]

    def objective(params: np.ndarray) -> float:
        transform = delta_pose(base_pose, params)
        projection = projected_bbox_mask(transform, corners, intrinsics, target.shape)
        iou = mask_iou(projection, target)
        motion_penalty = 0.03 * float(np.sum((params[:3] / max_translation_m) ** 2))
        rotation_penalty = 0.01 * float(np.sum((params[3:] / max_rot_rad) ** 2))
        return -iou + motion_penalty + rotation_penalty

    starts = [
        np.zeros(6, dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(8.0)]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-8.0)]),
    ]
    best_params = starts[0]
    best_value = objective(best_params)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 90, "xtol": 1e-4, "ftol": 1e-4, "disp": False},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_params = np.asarray(result.x, dtype=np.float64)
    refined = delta_pose(base_pose, best_params)
    refined_iou = mask_iou(projected_bbox_mask(refined, corners, intrinsics, target.shape), target)
    if refined_iou < base_iou:
        return base_pose, base_iou, base_iou
    return refined, base_iou, refined_iou


def write_pose_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    extras = ["image_fallback", "silhouette_iou_before", "silhouette_iou_after"]
    for field in extras:
        if field not in fieldnames:
            fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in extras:
                output.setdefault(field, "")
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair selected pose frames using bounded 2D bbox/SAM-mask silhouette overlap.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--mask_map", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True, help="Frame ids or ranges, e.g. 86-96 124-133.")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_translation_m", type=float, default=0.012)
    parser.add_argument("--max_z_m", type=float, default=0.003)
    parser.add_argument("--max_rotation_deg", type=float, default=18.0)
    parser.add_argument("--use_previous_refined", action="store_true")
    args = parser.parse_args()

    target_frames = parse_ranges(args.frames)
    rows, row_by_frame = load_pose_rows(args.poses)
    mask_paths = load_mask_map(args.mask_map)
    stl_extent = get_projectable_stl_bbox_extent(args.stl)
    corners = bbox_corners(stl_extent)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
    finally:
        pipeline.stop()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = args.out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    refined_by_frame: dict[int, np.ndarray] = {}
    diagnostics: list[dict[str, str]] = []

    for frame in iter_aligned_frames(args.bag):
        if frame.frame_id not in target_frames:
            continue
        row = row_by_frame.get(frame.frame_id)
        mask_path = mask_paths.get(frame.frame_id)
        if row is None or row.get("detected") != "1" or not row.get("tx") or mask_path is None:
            continue
        sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if sam_mask is None:
            continue
        if sam_mask.shape != frame.color_rgb.shape[:2]:
            sam_mask = cv2.resize(
                sam_mask,
                (frame.color_rgb.shape[1], frame.color_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        base_pose = row_to_transform(row)
        if args.use_previous_refined and (frame.frame_id - 1) in refined_by_frame:
            base_pose = refined_by_frame[frame.frame_id - 1]

        refined, iou_before, iou_after = optimize_to_mask(
            base_pose,
            sam_mask,
            corners,
            intrinsics,
            args.max_translation_m,
            args.max_z_m,
            args.max_rotation_deg,
        )
        refined_by_frame[frame.frame_id] = refined

        quat = matrix_to_quaternion_xyzw(refined)
        row["tx"] = f"{refined[0, 3]:.9f}"
        row["ty"] = f"{refined[1, 3]:.9f}"
        row["tz"] = f"{refined[2, 3]:.9f}"
        row["qx"] = f"{quat[0]:.9f}"
        row["qy"] = f"{quat[1]:.9f}"
        row["qz"] = f"{quat[2]:.9f}"
        row["qw"] = f"{quat[3]:.9f}"
        row["image_fallback"] = "1"
        row["silhouette_iou_before"] = f"{iou_before:.6f}"
        row["silhouette_iou_after"] = f"{iou_after:.6f}"

        image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        blue = np.zeros_like(image_bgr)
        blue[:, :, 0] = 255
        image_bgr = np.where(sam_mask[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.45, blue, 0.55, 0), image_bgr)
        base_rvec, base_tvec = matrix_to_rvec_tvec(base_pose)
        image_bgr = draw_projected_bbox(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            intrinsics,
            base_rvec,
            base_tvec,
            stl_extent,
            color=(0, 165, 255),
            thickness=2,
        )
        refined_rvec, refined_tvec = matrix_to_rvec_tvec(refined)
        image_bgr = draw_projected_bbox(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            intrinsics,
            refined_rvec,
            refined_tvec,
            stl_extent,
            color=(0, 255, 0),
            thickness=2,
        )
        text = f"frame {frame.frame_id} blue: SAM orange: before green: image fallback IoU {iou_before:.3f}->{iou_after:.3f}"
        cv2.putText(image_bgr, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlays_dir / f"frame_{frame.frame_id:06d}_silhouette_refine.png"), image_bgr)

        diagnostics.append(
            {
                "frame_id": str(frame.frame_id),
                "iou_before": f"{iou_before:.6f}",
                "iou_after": f"{iou_after:.6f}",
            }
        )

    write_pose_csv(args.out_csv, rows)
    diagnostics_path = args.out_dir / "silhouette_refine_summary.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "iou_before", "iou_after"])
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"refined {len(diagnostics)} frames")
    print(f"saved {args.out_csv}")
    print(f"saved overlays to {overlays_dir}")
    print(f"saved summary to {diagnostics_path}")


if __name__ == "__main__":
    main()
