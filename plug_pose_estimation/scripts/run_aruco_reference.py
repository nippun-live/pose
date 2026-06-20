from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.aruco_pose import detect_marker, estimate_marker_pose  # noqa: E402
from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.transforms import (  # noqa: E402
    matrix_to_quaternion_xyzw,
    matrix_to_rvec_tvec,
    rpy_deg_to_matrix,
    rvec_tvec_to_matrix,
)
from plug_pose.visualization import draw_marker_detection, draw_pose_axes, draw_projected_bbox  # noqa: E402


def load_marker_to_plug(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    marker_to_plug = config["marker_to_plug"]
    return rpy_deg_to_matrix(marker_to_plug["translation_m"], marker_to_plug["rotation_rpy_deg"])


def blank_pose_row(frame_id: int, timestamp_s: float, detected: bool) -> dict[str, str]:
    return {
        "frame_id": str(frame_id),
        "timestamp": f"{timestamp_s:.6f}",
        "detected": str(int(detected)),
        "tx": "",
        "ty": "",
        "tz": "",
        "qx": "",
        "qy": "",
        "qz": "",
        "qw": "",
    }


def fill_pose(row: dict[str, str], transform) -> dict[str, str]:
    quat = matrix_to_quaternion_xyzw(transform)
    row.update(
        {
            "tx": f"{transform[0, 3]:.9f}",
            "ty": f"{transform[1, 3]:.9f}",
            "tz": f"{transform[2, 3]:.9f}",
            "qx": f"{quat[0]:.9f}",
            "qy": f"{quat[1]:.9f}",
            "qz": f"{quat[2]:.9f}",
            "qw": f"{quat[3]:.9f}",
        }
    )
    return row


def write_pose_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_id", "timestamp", "detected", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ArUco marker pose and derived plug-frame reference pose.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--marker_length", type=float, default=0.011)
    parser.add_argument("--marker_id", type=int, default=32)
    parser.add_argument("--out_marker_csv", type=Path, required=True)
    parser.add_argument("--out_plug_csv", type=Path, required=True)
    parser.add_argument("--out_video", type=Path, required=True)
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    t_marker_plug = load_marker_to_plug(args.config)
    stl_extent = get_projectable_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    marker_rows: list[dict[str, str]] = []
    plug_rows: list[dict[str, str]] = []
    detections = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        corners, ids = detect_marker(frame.color_rgb, marker_id=args.marker_id)
        detected = corners is not None
        marker_row = blank_pose_row(frame.frame_id, frame.timestamp_s, detected)
        plug_row = blank_pose_row(frame.frame_id, frame.timestamp_s, detected)
        output_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)

        if detected:
            rvec_marker, tvec_marker = estimate_marker_pose(corners, intrinsics, args.marker_length)
            t_camera_marker = rvec_tvec_to_matrix(rvec_marker, tvec_marker)
            t_camera_plug = t_camera_marker @ t_marker_plug
            fill_pose(marker_row, t_camera_marker)
            fill_pose(plug_row, t_camera_plug)
            detections += 1

            output_bgr = draw_marker_detection(frame.color_rgb, corners, ids)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            output_bgr = draw_pose_axes(output_rgb, intrinsics, rvec_marker, tvec_marker, 0.01)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            rvec_plug, tvec_plug = matrix_to_rvec_tvec(t_camera_plug)
            output_bgr = draw_projected_bbox(
                output_rgb,
                intrinsics,
                rvec_plug,
                tvec_plug,
                stl_extent,
                color=(0, 255, 0),
                thickness=2,
            )

        if writer is None:
            height, width = output_bgr.shape[:2]
            writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        writer.write(output_bgr)
        marker_rows.append(marker_row)
        plug_rows.append(plug_row)

    if writer is not None:
        writer.release()
    write_pose_csv(args.out_marker_csv, marker_rows)
    write_pose_csv(args.out_plug_csv, plug_rows)

    print(f"processed {len(marker_rows)} frames")
    print(f"detected marker {args.marker_id} in {detections} frames")
    print(f"saved marker poses to {args.out_marker_csv}")
    print(f"saved plug reference poses to {args.out_plug_csv}")
    print(f"saved overlay video to {args.out_video}")


if __name__ == "__main__":
    main()
