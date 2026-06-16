from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_color_intrinsics, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.project_mesh import get_projectable_stl_bbox_extent  # noqa: E402
from plug_pose.transforms import matrix_to_rvec_tvec, pose_quat_xyzw_to_matrix  # noqa: E402
from plug_pose.visualization import draw_projected_bbox  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle)}


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def draw_box_bgr(image_bgr: np.ndarray, intrinsics: dict, transform: np.ndarray, extent, color) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rvec, tvec = matrix_to_rvec_tvec(transform)
    boxed = draw_projected_bbox(image_rgb, intrinsics, rvec, tvec, extent, color=color, thickness=2)
    return boxed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--init_csv", type=Path, required=True)
    parser.add_argument("--icp_csv", type=Path, required=True)
    parser.add_argument("--reference_csv", type=Path)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--out_image", type=Path, required=True)
    args = parser.parse_args()

    init_poses = load_pose_csv(args.init_csv)
    icp_poses = load_pose_csv(args.icp_csv)
    ref_poses = load_pose_csv(args.reference_csv) if args.reference_csv else {}

    if args.frame_id not in init_poses:
        raise RuntimeError(f"frame {args.frame_id} missing from init CSV")
    if args.frame_id not in icp_poses:
        raise RuntimeError(f"frame {args.frame_id} missing from ICP CSV")

    init_t = row_to_transform(init_poses[args.frame_id])
    icp_t = row_to_transform(icp_poses[args.frame_id])
    ref_t = row_to_transform(ref_poses[args.frame_id]) if args.frame_id in ref_poses else None

    pipeline, profile = start_playback(args.bag)
    try:
        intrinsics = get_color_intrinsics(profile)
    finally:
        pipeline.stop()

    frame = None
    for candidate in iter_aligned_frames(args.bag, max_frames=args.frame_id + 1):
        frame = candidate
    if frame is None or frame.frame_id != args.frame_id:
        raise RuntimeError(f"Could not read frame {args.frame_id}")

    extent = get_projectable_stl_bbox_extent(args.stl)
    image = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    image = draw_box_bgr(image, intrinsics, init_t, extent, (0, 165, 255))
    image = draw_box_bgr(image, intrinsics, icp_t, extent, (0, 255, 0))
    if ref_t is not None:
        image = draw_box_bgr(image, intrinsics, ref_t, extent, (255, 0, 255))

    cv2.putText(image, "orange:init green:icp magenta:reference", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_image), image)

    print(f"frame_id: {args.frame_id}")
    print(f"init -> icp translation delta m: {np.linalg.norm(icp_t[:3, 3] - init_t[:3, 3]):.9f}")
    print(f"init -> icp rotation delta deg: {rotation_error_deg(init_t, icp_t):.6f}")
    if ref_t is not None:
        print(f"reference -> init translation error m: {np.linalg.norm(init_t[:3, 3] - ref_t[:3, 3]):.9f}")
        print(f"reference -> init rotation error deg: {rotation_error_deg(ref_t, init_t):.6f}")
        print(f"reference -> icp translation error m: {np.linalg.norm(icp_t[:3, 3] - ref_t[:3, 3]):.9f}")
        print(f"reference -> icp rotation error deg: {rotation_error_deg(ref_t, icp_t):.6f}")
    print(f"saved overlay image to {args.out_image}")


if __name__ == "__main__":
    main()
