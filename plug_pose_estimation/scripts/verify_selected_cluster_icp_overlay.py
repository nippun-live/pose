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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--init_csv", type=Path, required=True)
    parser.add_argument("--icp_csv", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--out_image", type=Path, required=True)
    args = parser.parse_args()

    init_rows = load_pose_csv(args.init_csv)
    icp_rows = load_pose_csv(args.icp_csv)
    if args.frame_id not in init_rows:
        raise RuntimeError(f"frame {args.frame_id} missing from init CSV")
    if args.frame_id not in icp_rows:
        raise RuntimeError(f"frame {args.frame_id} missing from ICP CSV")

    init_t = row_to_transform(init_rows[args.frame_id])
    icp_t = row_to_transform(icp_rows[args.frame_id])

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

    image_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {args.mask}")
    if mask.shape[:2] != image_bgr.shape[:2]:
        raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {image_bgr.shape[:2]}")

    blue = np.zeros_like(image_bgr)
    blue[:, :, 0] = 255
    masked = np.where(mask[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.45, blue, 0.55, 0), image_bgr)

    extent = get_projectable_stl_bbox_extent(args.stl)
    init_rvec, init_tvec = matrix_to_rvec_tvec(init_t)
    init_box_bgr = draw_projected_bbox(
        cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
        intrinsics,
        init_rvec,
        init_tvec,
        extent,
        color=(0, 165, 255),
        thickness=2,
    )
    init_box_rgb = cv2.cvtColor(init_box_bgr, cv2.COLOR_BGR2RGB)

    icp_rvec, icp_tvec = matrix_to_rvec_tvec(icp_t)
    both_bgr = draw_projected_bbox(
        init_box_rgb,
        intrinsics,
        icp_rvec,
        icp_tvec,
        extent,
        color=(0, 255, 0),
        thickness=2,
    )

    cv2.putText(
        both_bgr,
        "blue:selected depth  orange:before ICP  green:after ICP",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        both_bgr,
        "blue:selected depth  orange:before ICP  green:after ICP",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_image), both_bgr)
    print(f"saved overlay image to {args.out_image}")


if __name__ == "__main__":
    main()
