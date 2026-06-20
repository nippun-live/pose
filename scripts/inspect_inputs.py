from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, get_depth_scale, start_playback, stream_summaries  # noqa: E402


def inspect_bag(path: Path) -> dict:
    pipeline, profile = start_playback(path)
    try:
        return {
            "file": str(path),
            "streams": [summary.__dict__ for summary in stream_summaries(profile)],
            "color_intrinsics": get_color_intrinsics(profile),
            "depth_scale_m_per_unit": get_depth_scale(profile),
        }
    finally:
        pipeline.stop()


def inspect_stl(path: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load STL: {path}")
    bbox = mesh.get_axis_aligned_bounding_box()
    return {
        "file": str(path),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "bbox_min": bbox.get_min_bound().tolist(),
        "bbox_max": bbox.get_max_bound().tolist(),
        "bbox_extent": bbox.get_extent().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect input RealSense bags and STL model.")
    parser.add_argument("--bag", type=Path, action="append", default=[])
    parser.add_argument("--stl", type=Path)
    parser.add_argument("--out_json", type=Path)
    args = parser.parse_args()

    report = {"bags": [inspect_bag(path) for path in args.bag]}
    if args.stl:
        report["stl"] = inspect_stl(args.stl)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
