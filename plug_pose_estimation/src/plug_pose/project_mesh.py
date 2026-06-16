from __future__ import annotations

from pathlib import Path

from plug_pose.stl_utils import get_centered_stl_bbox_extent


def get_projectable_stl_bbox_extent(stl_path: str | Path):
    return get_centered_stl_bbox_extent(stl_path)
