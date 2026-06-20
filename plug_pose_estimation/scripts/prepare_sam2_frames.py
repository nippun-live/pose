from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, iter_aligned_frames, save_intrinsics, start_playback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    rgb_dir = args.out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
    finally:
        pipeline.stop()
    save_intrinsics(args.out_dir / "intrinsics.json", intrinsics, depth_scale)

    frame_map = []
    end_frame = None if args.max_frames is None else args.start_frame + args.max_frames
    for frame in iter_aligned_frames(args.bag, max_frames=end_frame):
        if frame.frame_id < args.start_frame:
            continue
        sam_idx = len(frame_map)
        bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        out_path = rgb_dir / f"{sam_idx:05d}.jpg"
        cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
        frame_map.append(
            {
                "sam_frame_idx": sam_idx,
                "bag_frame_id": frame.frame_id,
                "timestamp_s": frame.timestamp_s,
                "rgb": str(out_path.as_posix()),
            }
        )

    (args.out_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    print(f"saved {len(frame_map)} JPEG frames to {rgb_dir}")
    print(f"saved frame map to {args.out_dir / 'frame_map.json'}")
    print(f"saved intrinsics to {args.out_dir / 'intrinsics.json'}")


if __name__ == "__main__":
    main()
