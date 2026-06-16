from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import iter_aligned_frames  # noqa: E402
from plug_pose.visualization import colorize_depth  # noqa: E402


def resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--thumb_width", type=int, default=240)
    parser.add_argument("--sheet_cols", type=int, default=4)
    parser.add_argument("--frames_per_sheet", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    sheets_dir = args.out_dir / "sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[np.ndarray] = []
    saved = 0

    for frame in iter_aligned_frames(args.bag, max_frames=args.max_frames):
        if frame.frame_id % args.stride != 0:
            continue

        color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        depth_vis = colorize_depth(frame.depth_z16)
        color_small = resize_keep_aspect(color_bgr, args.thumb_width)
        depth_small = resize_keep_aspect(depth_vis, args.thumb_width)
        tile = np.hstack([color_small, depth_small])
        label = f"frame {frame.frame_id:06d}  t={frame.timestamp_s:.2f}s"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(frames_dir / f"frame_{frame.frame_id:06d}_color_depth.png"), tile)
        tiles.append(tile)
        saved += 1

        if len(tiles) == args.frames_per_sheet:
            sheet_idx = math.ceil(saved / args.frames_per_sheet) - 1
            save_sheet(tiles, args.sheet_cols, sheets_dir / f"sheet_{sheet_idx:03d}.png")
            tiles = []

    if tiles:
        sheet_idx = math.ceil(saved / args.frames_per_sheet) - 1
        save_sheet(tiles, args.sheet_cols, sheets_dir / f"sheet_{sheet_idx:03d}.png")

    print(f"saved {saved} color/depth frame review images")
    print(f"per-frame images: {frames_dir}")
    print(f"contact sheets: {sheets_dir}")


def save_sheet(tiles: list[np.ndarray], cols: int, out_path: Path) -> None:
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 24, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    cv2.imwrite(str(out_path), sheet)


if __name__ == "__main__":
    main()
