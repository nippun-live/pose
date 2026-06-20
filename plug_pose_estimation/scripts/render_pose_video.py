from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, intrinsics_to_camera_matrix, intrinsics_to_dist_coeffs, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.transforms import matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            int(row["frame_id"]): row
            for row in csv.DictReader(handle)
            if row.get("detected", "1") == "1" and row.get("tx")
        }


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def draw_colored_axes(
    image_bgr: np.ndarray,
    intrinsics: dict,
    transform: np.ndarray,
    axis_length_m: float,
    colors_bgr: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    thickness: int = 3,
) -> None:
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float32,
    )
    rvec, tvec = matrix_to_rvec_tvec(transform)
    projected, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        intrinsics_to_camera_matrix(intrinsics),
        intrinsics_to_dist_coeffs(intrinsics),
    )
    points = projected.reshape(-1, 2)
    if not np.isfinite(points).all():
        return
    points = np.round(points).astype(np.int32)
    h, w = image_bgr.shape[:2]
    origin = points[0]
    if origin[0] < -w or origin[0] > 2 * w or origin[1] < -h or origin[1] > 2 * h:
        return
    for idx, color in zip([1, 2, 3], colors_bgr):
        end = points[idx]
        cv2.line(
            image_bgr,
            (int(origin[0]), int(origin[1])),
            (int(end[0]), int(end[1])),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_center_marker(
    image_bgr: np.ndarray,
    intrinsics: dict,
    transform: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> tuple[int, int] | None:
    rvec, tvec = matrix_to_rvec_tvec(transform)
    projected, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        rvec,
        tvec,
        intrinsics_to_camera_matrix(intrinsics),
        intrinsics_to_dist_coeffs(intrinsics),
    )
    point = projected.reshape(2)
    if not np.isfinite(point).all():
        return None
    xy = tuple(np.round(point).astype(int).tolist())
    cv2.circle(image_bgr, xy, 5, color_bgr, -1, cv2.LINE_AA)
    return xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one or two pose CSVs as bbox + pose-axis video.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--pose_a", type=Path, required=True)
    parser.add_argument("--pose_b", type=Path)
    parser.add_argument("--label_a", default="pose A")
    parser.add_argument("--label_b", default="pose B")
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--out_video", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--axis_length_m", type=float, default=0.018)
    args = parser.parse_args()

    poses_a = load_pose_csv(args.pose_a)
    poses_b = load_pose_csv(args.pose_b) if args.pose_b else {}
    bbox_extent = get_projectable_stl_bbox_extent(args.stl)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    processed = 0
    drawn = 0
    end_frame = None if args.max_frames is None else args.start_frame + args.max_frames

    for frame in iter_aligned_frames(args.bag, max_frames=end_frame):
        if frame.frame_id < args.start_frame:
            continue
        image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)

        row_a = poses_a.get(frame.frame_id)
        row_b = poses_b.get(frame.frame_id)
        center_a = None
        center_b = None
        if row_a is not None:
            transform_a = row_to_transform(row_a)
            rvec_a, tvec_a = matrix_to_rvec_tvec(transform_a)
            bbox_bgr = draw_projected_bbox(frame.color_rgb, intrinsics, rvec_a, tvec_a, bbox_extent, color=(0, 255, 0), thickness=2)
            image_bgr = bbox_bgr
            draw_colored_axes(
                image_bgr,
                intrinsics,
                transform_a,
                args.axis_length_m,
                colors_bgr=((0, 180, 0), (0, 255, 160), (0, 255, 255)),
            )
            center_a = draw_center_marker(image_bgr, intrinsics, transform_a, (0, 255, 0))
            drawn += 1
        if row_b is not None:
            transform_b = row_to_transform(row_b)
            overlay_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rvec_b, tvec_b = matrix_to_rvec_tvec(transform_b)
            image_bgr = draw_projected_bbox(overlay_rgb, intrinsics, rvec_b, tvec_b, bbox_extent, color=(255, 0, 255), thickness=2)
            draw_colored_axes(
                image_bgr,
                intrinsics,
                transform_b,
                args.axis_length_m,
                colors_bgr=((180, 0, 180), (255, 120, 255), (255, 255, 255)),
            )
            center_b = draw_center_marker(image_bgr, intrinsics, transform_b, (255, 0, 255))

        if center_a is not None and center_b is not None:
            cv2.line(image_bgr, center_a, center_b, (255, 255, 255), 1, cv2.LINE_AA)

        text = f"green: {args.label_a}"
        if args.pose_b:
            text += f"   magenta: {args.label_b}"
        cv2.putText(image_bgr, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            h, w = image_bgr.shape[:2]
            writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {args.out_video}")
        writer.write(image_bgr)
        processed += 1

    if writer is not None:
        writer.release()
    if not args.out_video.exists() or args.out_video.stat().st_size == 0:
        raise RuntimeError(f"Video was not written: {args.out_video}")
    print(f"processed {processed} frames")
    print(f"drew pose A on {drawn} frames")
    print(f"saved {args.out_video} ({args.out_video.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
