from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.aruco_pose import detect_marker  # noqa: E402
from plug_pose.visualization import draw_marker_detection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--marker_id", type=int, default=32)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    color_paths = sorted(args.frames.glob("color_*.png"))
    detections = 0

    for path in color_paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        corners, ids = detect_marker(image_rgb, marker_id=args.marker_id)
        detections += int(corners is not None)
        output = draw_marker_detection(image_rgb, corners, ids)
        frame_id = path.stem.split("_")[-1]
        cv2.imwrite(str(args.out / f"aruco_{frame_id}.png"), output)

    print(f"processed {len(color_paths)} frames")
    print(f"detected marker {args.marker_id} in {detections} frames")
    print(f"saved debug images to {args.out}")


if __name__ == "__main__":
    main()

