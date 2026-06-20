from __future__ import annotations

from pathlib import Path

import open3d as o3d


def load_centered_stl_mesh(stl_path: str | Path) -> tuple[o3d.geometry.TriangleMesh, object]:
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh: {stl_path}")
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    mesh.compute_vertex_normals()
    return mesh, bbox


def load_stl_as_pointcloud(stl_path: str | Path, n_points: int = 5000) -> o3d.geometry.PointCloud:
    mesh, _bbox = load_centered_stl_mesh(stl_path)
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    return pcd


def get_centered_stl_bbox_extent(stl_path: str | Path):
    mesh, _bbox = load_centered_stl_mesh(stl_path)
    return mesh.get_axis_aligned_bounding_box().get_extent()
