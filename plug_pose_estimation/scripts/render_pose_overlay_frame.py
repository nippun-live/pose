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
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--out_png", type=Path, required=True)
    parser.add_argument("--overlay", choices=["bbox", "mesh", "both"], default="both")
    parser.add_argument("--mesh_max_triangles", type=int, default=500)
    args = parser.parse_args()

    poses = load_pose_csv(args.poses)
    bbox_extent = get_projectable_stl_bbox_extent(args.stl)
    mesh_vertices, mesh_edges = get_projectable_stl_wireframe(args.stl, args.mesh_max_triangles)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
    finally:
        pipeline.stop()

    pose = poses.get(args.frame_id)
    if pose is None or pose.get("detected") != "1":
        raise ValueError(f"No detected pose for frame {args.frame_id}")

    transform = pose_quat_xyzw_to_matrix(
        [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])],
        [float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])],
    )
    rvec, tvec = matrix_to_rvec_tvec(transform)

    for frame in iter_aligned_frames(args.bag, max_frames=args.frame_id + 1):
        if frame.frame_id != args.frame_id:
            continue
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
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_png), output_bgr)
        print(f"saved {args.out_png}")
        return
    raise ValueError(f"Frame {args.frame_id} not found")


if __name__ == "__main__":
    main()
