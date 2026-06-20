from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import (  # noqa: E402
    get_color_intrinsics,
    get_depth_scale,
    iter_aligned_frames,
    save_intrinsics,
    start_playback,
    stream_summaries,
)
from plug_pose.visualization import colorize_depth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare aligned RealSense frames for inspection and SAM2 segmentation."
    )
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--save_rgbd_samples", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "rgbd_samples"
    if args.save_rgbd_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
        summaries = stream_summaries(profile)
    finally:
        pipeline.stop()

    save_intrinsics(args.out_dir / "intrinsics.json", intrinsics, depth_scale)
    (args.out_dir / "streams.json").write_text(
        json.dumps([summary.__dict__ for summary in summaries], indent=2),
        encoding="utf-8",
    )

    frame_map = []
    end_frame = None if args.max_frames is None else args.start_frame + args.max_frames
    for frame in iter_aligned_frames(args.bag, max_frames=end_frame):
        if frame.frame_id < args.start_frame:
            continue
        sam_idx = len(frame_map)
        bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        jpg_path = rgb_dir / f"{sam_idx:05d}.jpg"
        cv2.imwrite(str(jpg_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])

        if args.save_rgbd_samples:
            cv2.imwrite(str(samples_dir / f"color_{frame.frame_id:06d}.png"), bgr)
            np.save(samples_dir / f"depth_{frame.frame_id:06d}.npy", frame.depth_z16)
            cv2.imwrite(str(samples_dir / f"depth_vis_{frame.frame_id:06d}.png"), colorize_depth(frame.depth_z16))

        frame_map.append(
            {
                "sam_frame_idx": sam_idx,
                "bag_frame_id": frame.frame_id,
                "timestamp_s": frame.timestamp_s,
                "rgb": str(jpg_path.as_posix()),
            }
        )

    (args.out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    print(f"saved {len(frame_map)} SAM2 JPEG frames to {rgb_dir}")
    print(f"saved frame map to {args.out_dir / 'frame_map.json'}")
    print(f"saved intrinsics to {args.out_dir / 'intrinsics.json'}")


if __name__ == "__main__":
    main()
