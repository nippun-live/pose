from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.transforms import matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix, rpy_deg_to_matrix  # noqa: E402
from plug_pose.visualization import draw_pose_axes, draw_projected_bbox  # noqa: E402


def load_marker_to_plug(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    marker_to_plug = config["marker_to_plug"]
    return rpy_deg_to_matrix(
        marker_to_plug["translation_m"],
        marker_to_plug["rotation_rpy_deg"],
    )


def load_marker_poses(csv_path: Path) -> dict[int, dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {int(row["frame_id"]): row for row in reader}


def load_stl_bbox_extent(stl_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh: {stl_path}")
    return mesh.get_axis_aligned_bounding_box().get_extent()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--marker_csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--out_video", type=Path, required=True)
    parser.add_argument("--axis_length", type=float, default=0.02)
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    marker_poses = load_marker_poses(args.marker_csv)
    t_marker_plug = load_marker_to_plug(args.config)
    bbox_extent = load_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    detected_count = 0
    frame_count = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        marker_row = marker_poses.get(frame.frame_id)
        detected = marker_row is not None and marker_row["detected"] == "1"
        output_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)

        if detected:
            t_camera_marker = pose_quat_xyzw_to_matrix(
                [float(marker_row["tx"]), float(marker_row["ty"]), float(marker_row["tz"])],
                [float(marker_row["qx"]), float(marker_row["qy"]), float(marker_row["qz"]), float(marker_row["qw"])],
            )
            t_camera_plug = t_camera_marker @ t_marker_plug
            rvec, tvec = matrix_to_rvec_tvec(t_camera_plug)
            bbox_bgr = draw_projected_bbox(frame.color_rgb, intrinsics, rvec, tvec, bbox_extent)
            bbox_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)
            output_bgr = draw_pose_axes(bbox_rgb, intrinsics, rvec, tvec, args.axis_length)
            detected_count += 1

        if writer is None:
            height, width = output_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.out_video), fourcc, float(fps), (width, height))
        writer.write(output_bgr)
        frame_count += 1

    if writer is not None:
        writer.release()

    print(f"processed {frame_count} frames")
    print(f"drew STL bounding box for {detected_count} frames")
    print(f"bbox extent m: {bbox_extent}")
    print(f"saved overlay video to {args.out_video}")


if __name__ == "__main__":
    main()
