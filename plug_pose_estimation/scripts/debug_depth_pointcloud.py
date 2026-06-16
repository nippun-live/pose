from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.pointcloud import backproject_depth_roi, downsample_cloud  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--roi_key", default="roi_tagged")
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    markerless = config["markerless"]
    roi = markerless[args.roi_key]
    depth_config = markerless["depth"]
    voxel_size = markerless["icp"]["voxel_size_m"]

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
    finally:
        pipeline.stop()

    if depth_scale is None:
        raise RuntimeError("No depth scale found in bag.")

    selected_frame = None
    for frame in iter_aligned_frames(args.bag, max_frames=args.frame_id + 1):
        selected_frame = frame
    if selected_frame is None or selected_frame.frame_id != args.frame_id:
        raise RuntimeError(f"Could not read frame {args.frame_id}")

    pcd = backproject_depth_roi(
        selected_frame.depth_z16,
        selected_frame.color_rgb,
        intrinsics,
        depth_scale,
        roi,
        depth_config["min_m"],
        depth_config["max_m"],
    )
    pcd = downsample_cloud(pcd, voxel_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pcd)
    print(f"frame_id: {selected_frame.frame_id}")
    print(f"timestamp: {selected_frame.timestamp_s:.6f}")
    print(f"points: {len(pcd.points)}")
    print(f"saved observed point cloud to {args.out}")

    if args.view:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
