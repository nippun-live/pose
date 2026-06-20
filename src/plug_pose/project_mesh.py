from __future__ import annotations

from pathlib import Path

import numpy as np

from plug_pose.stl_utils import get_centered_stl_bbox_extent
from plug_pose.stl_utils import load_centered_stl_mesh


def get_projectable_stl_bbox_extent(stl_path: str | Path):
    return get_centered_stl_bbox_extent(stl_path)


def get_projectable_stl_wireframe(stl_path: str | Path, max_triangles: int = 2000):
    mesh, _bbox = load_centered_stl_mesh(stl_path)
    if max_triangles > 0 and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    edge_set = set()
    for tri in triangles:
        a, b, c = [int(v) for v in tri]
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((b, c))))
        edge_set.add(tuple(sorted((c, a))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    return vertices, edges
