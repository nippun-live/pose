from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def format_mesh_summary(path: Path) -> str:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = bbox.get_extent()

    lines = [
        f"filename: {path}",
        f"vertices: {len(mesh.vertices)}",
        f"triangles: {len(mesh.triangles)}",
        f"bbox min: [{min_bound[0]:.6f}, {min_bound[1]:.6f}, {min_bound[2]:.6f}]",
        f"bbox max: [{max_bound[0]:.6f}, {max_bound[1]:.6f}, {max_bound[2]:.6f}]",
        f"bbox extent: [{extent[0]:.6f}, {extent[1]:.6f}, {extent[2]:.6f}]",
        "scale note: extents near 20,10,5 imply millimeters; extents near 0.020,0.010,0.005 imply meters.",
        "canonical frame note: STL coordinates are treated as the canonical plug frame.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stl", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    summary = format_mesh_summary(args.stl)
    print(summary, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")

    if not args.no_view:
        mesh = o3d.io.read_triangle_mesh(str(args.stl))
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()

