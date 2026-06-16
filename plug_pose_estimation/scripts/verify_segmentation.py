from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.transforms import pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import colorize_depth  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--roi_key", default="roi_tagged")
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--pose_csv", type=Path)
    parser.add_argument("--stl", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    markerless = config["markerless"]
    roi = markerless[args.roi_key]
    depth_cfg = markerless["depth"]

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
    finally:
        pipeline.stop()
    if depth_scale is None:
        raise RuntimeError("No depth scale found in bag.")

    frame = None
    for candidate in iter_aligned_frames(args.bag, max_frames=args.frame_id + 1):
        frame = candidate
    if frame is None or frame.frame_id != args.frame_id:
        raise RuntimeError(f"Could not read frame {args.frame_id}")

    x_min = max(0, int(roi["x_min"]))
    y_min = max(0, int(roi["y_min"]))
    x_max = min(frame.depth_z16.shape[1], int(roi["x_max"]))
    y_max = min(frame.depth_z16.shape[0], int(roi["y_max"]))

    depth_m = frame.depth_z16.astype(np.float64) * depth_scale
    mask = np.zeros(frame.depth_z16.shape, dtype=np.uint8)
    valid_roi = (
        (depth_m[y_min:y_max, x_min:x_max] > depth_cfg["min_m"])
        & (depth_m[y_min:y_max, x_min:x_max] < depth_cfg["max_m"])
    )
    mask[y_min:y_max, x_min:x_max][valid_roi] = 255

    color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    roi_bgr = color_bgr.copy()
    cv2.rectangle(roi_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    overlay = color_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.45, red, 0.55, 0), overlay)
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    depth_vis = colorize_depth(frame.depth_z16)
    depth_overlay = depth_vis.copy()
    depth_overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(depth_overlay, 0.45, red, 0.55, 0), depth_overlay)
    cv2.rectangle(depth_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_roi.png"), roi_bgr)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_segmentation_rgb.png"), overlay)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_segmentation_depth.png"), depth_overlay)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_mask.png"), mask)

    pose_gate_pixels = None
    if args.pose_csv and args.stl:
        poses = load_pose_csv(args.pose_csv)
        pose_row = poses.get(frame.frame_id)
        if pose_row is not None and pose_row["detected"] == "1":
            transform = row_to_transform(pose_row)
            extent = get_projectable_stl_bbox_extent(args.stl)
            margin = markerless["icp"].get("pose_gate_margin_m", 0.0)

            ys, xs = np.nonzero(mask > 0)
            z = depth_m[ys, xs]
            x = (xs - intrinsics["ppx"]) * z / intrinsics["fx"]
            y = (ys - intrinsics["ppy"]) * z / intrinsics["fy"]
            points = np.column_stack((x, y, z))
            local = (transform[:3, :3].T @ (points - transform[:3, 3]).T).T
            half_extent = np.asarray(extent, dtype=np.float64) / 2.0 + float(margin)
            keep = np.all(np.abs(local) <= half_extent, axis=1)

            pose_mask = np.zeros_like(mask)
            pose_mask[ys[keep], xs[keep]] = 255
            pose_gate_pixels = int(np.count_nonzero(pose_mask))

            pose_overlay = color_bgr.copy()
            green = np.zeros_like(pose_overlay)
            green[:, :, 1] = 255
            pose_overlay = np.where(pose_mask[:, :, None] > 0, cv2.addWeighted(pose_overlay, 0.4, green, 0.6, 0), pose_overlay)
            cv2.rectangle(pose_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_pose_gated_rgb.png"), pose_overlay)
            cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_pose_gated_mask.png"), pose_mask)

    print(f"frame_id: {frame.frame_id}")
    print(f"roi: x={x_min}:{x_max}, y={y_min}:{y_max}")
    print(f"depth range m: {depth_cfg['min_m']} to {depth_cfg['max_m']}")
    print(f"segmented pixels: {int(np.count_nonzero(mask))}")
    if pose_gate_pixels is not None:
        print(f"pose-gated pixels: {pose_gate_pixels}")
    print(f"saved verification images to {args.out_dir}")


if __name__ == "__main__":
    main()
