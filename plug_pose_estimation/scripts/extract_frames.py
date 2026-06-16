from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, save_intrinsics, start_playback  # noqa: E402
from plug_pose.visualization import colorize_depth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
    finally:
        pipeline.stop()

    save_intrinsics(args.out / "intrinsics.json", intrinsics, depth_scale)

    count = 0
    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out / f"color_{frame.frame_id:06d}.png"), color_bgr)
        np.save(args.out / f"depth_{frame.frame_id:06d}.npy", frame.depth_z16)
        cv2.imwrite(str(args.out / f"depth_vis_{frame.frame_id:06d}.png"), colorize_depth(frame.depth_z16))
        count += 1

    print(f"saved {count} aligned RGB-D frames to {args.out}")
    print(f"saved intrinsics to {args.out / 'intrinsics.json'}")


if __name__ == "__main__":
    main()

