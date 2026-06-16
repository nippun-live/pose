from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import (  # noqa: E402
    get_color_intrinsics,
    get_depth_scale,
    iter_aligned_frames,
    start_playback,
    stream_summaries,
)


def format_summary(bag_path: Path, scan_frames: bool = True) -> str:
    pipeline, profile = start_playback(bag_path)
    try:
        summaries = stream_summaries(profile)
        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
        playback = profile.get_device().as_playback()
        duration = playback.get_duration()
    finally:
        pipeline.stop()

    lines = [f"filename: {bag_path}"]
    lines.append("available streams:")
    for summary in summaries:
        lines.append(
            f"  - {summary.stream}[{summary.index}]: "
            f"{summary.width}x{summary.height} {summary.fps} FPS {summary.format}"
        )
    lines.append("color camera intrinsics:")
    lines.append(f"  width: {intrinsics['width']}")
    lines.append(f"  height: {intrinsics['height']}")
    lines.append(f"  fx: {intrinsics['fx']:.6f}")
    lines.append(f"  fy: {intrinsics['fy']:.6f}")
    lines.append(f"  ppx: {intrinsics['ppx']:.6f}")
    lines.append(f"  ppy: {intrinsics['ppy']:.6f}")
    lines.append(f"  model: {intrinsics['model']}")
    lines.append(f"  coeffs: {intrinsics['coeffs']}")
    lines.append(f"depth scale: {depth_scale} m/unit")
    lines.append(f"duration estimate from bag metadata: {duration.total_seconds():.3f} s")

    if scan_frames:
        frame_count = 0
        first_ts = None
        last_ts = None
        for frame in iter_aligned_frames(bag_path):
            frame_count += 1
            first_ts = frame.timestamp_s if first_ts is None else first_ts
            last_ts = frame.timestamp_s
        lines.append(f"aligned RGB-D frame count: {frame_count}")
        if first_ts is not None and last_ts is not None:
            lines.append(f"timestamp span: {last_ts - first_ts:.3f} s")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-scan", action="store_true", help="Skip full frame scan.")
    args = parser.parse_args()

    summary = format_summary(args.bag, scan_frames=not args.no_scan)
    print(summary, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()

