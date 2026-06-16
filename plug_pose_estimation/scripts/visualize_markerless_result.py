from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent, get_projectable_stl_wireframe  # noqa: E402
from plug_pose.transforms import matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_pose_axes, draw_projected_bbox, draw_projected_wireframe  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--out_video", type=Path, required=True)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--overlay", choices=["bbox", "mesh", "both"], default="bbox")
    parser.add_argument("--mesh_max_triangles", type=int, default=500)
    args = parser.parse_args()

    poses = load_pose_csv(args.poses)
    bbox_extent = get_projectable_stl_bbox_extent(args.stl)
    mesh_vertices, mesh_edges = get_projectable_stl_wireframe(args.stl, args.mesh_max_triangles)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        fps = profile.get_streams()[0].fps() or 15
    finally:
        pipeline.stop()

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    drawn = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        pose = poses.get(frame.frame_id)
        output_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        if pose is not None and pose["detected"] == "1":
            transform = pose_quat_xyzw_to_matrix(
                [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])],
                [float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])],
            )
            rvec, tvec = matrix_to_rvec_tvec(transform)
            overlay_rgb = frame.color_rgb
            if args.overlay in {"bbox", "both"}:
                bbox_bgr = draw_projected_bbox(overlay_rgb, intrinsics, rvec, tvec, bbox_extent, color=(0, 165, 255))
                overlay_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)
            if args.overlay in {"mesh", "both"}:
                mesh_bgr = draw_projected_wireframe(
                    overlay_rgb,
                    intrinsics,
                    rvec,
                    tvec,
                    mesh_vertices,
                    mesh_edges,
                    color=(0, 255, 0),
                    thickness=1,
                )
                overlay_rgb = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2RGB)
            output_bgr = draw_pose_axes(overlay_rgb, intrinsics, rvec, tvec, 0.02)
            drawn += 1

        if writer is None:
            height, width = output_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.out_video), fourcc, float(fps), (width, height))
        writer.write(output_bgr)
        count += 1

    if writer is not None:
        writer.release()

    print(f"processed {count} frames")
    print(f"drew markerless result for {drawn} frames")
    print(f"saved overlay video to {args.out_video}")


if __name__ == "__main__":
    main()
