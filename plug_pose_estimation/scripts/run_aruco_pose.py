from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.aruco_pose import detect_marker, estimate_marker_pose  # noqa: E402
from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.transforms import matrix_to_quaternion_xyzw, rvec_tvec_to_matrix  # noqa: E402
from plug_pose.visualization import draw_marker_detection, draw_pose_axes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--marker_length", type=float, default=0.011)
    parser.add_argument("--marker_id", type=int, default=32)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_video", type=Path, required=True)
    parser.add_argument("--axis_length", type=float, default=0.01)
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_video.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    detections = 0
    rows = []

    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        corners, ids = detect_marker(frame.color_rgb, marker_id=args.marker_id)
        detected = corners is not None

        row = {
            "frame_id": frame.frame_id,
            "timestamp": f"{frame.timestamp_s:.6f}",
            "detected": int(detected),
            "tx": "",
            "ty": "",
            "tz": "",
            "qx": "",
            "qy": "",
            "qz": "",
            "qw": "",
        }

        if detected:
            rvec, tvec = estimate_marker_pose(corners, intrinsics, args.marker_length)
            transform = rvec_tvec_to_matrix(rvec, tvec)
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
            output_bgr = draw_marker_detection(frame.color_rgb, corners, ids)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            output_bgr = draw_pose_axes(output_rgb, intrinsics, rvec, tvec, args.axis_length)
            detections += 1
        else:
            output_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)

        if writer is None:
            height, width = output_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.out_video), fourcc, float(fps), (width, height))
        writer.write(output_bgr)
        rows.append(row)

    if writer is not None:
        writer.release()

    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame_id", "timestamp", "detected", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print(f"processed {len(rows)} frames")
    print(f"detected marker {args.marker_id} in {detections} frames")
    print(f"saved poses to {args.out_csv}")
    print(f"saved overlay video to {args.out_video}")


if __name__ == "__main__":
    main()

