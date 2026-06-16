from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.bag_reader import get_depth_scale, iter_aligned_frames, start_playback  # noqa: E402
from plug_pose.visualization import colorize_depth  # noqa: E402


PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
    ],
    dtype=np.uint8,
)


def load_frame(bag: Path, frame_id: int):
    frame = None
    for candidate in iter_aligned_frames(bag, max_frames=frame_id + 1):
        frame = candidate
    if frame is None or frame.frame_id != frame_id:
        raise RuntimeError(f"Could not read frame {frame_id}")
    return frame


def backproject_with_pixels(depth_z16, color_rgb, intrinsics, depth_scale, roi, depth_cfg, color_cfg):
    x_min = max(0, int(roi["x_min"]))
    y_min = max(0, int(roi["y_min"]))
    x_max = min(depth_z16.shape[1], int(roi["x_max"]))
    y_max = min(depth_z16.shape[0], int(roi["y_max"]))

    depth_m = depth_z16.astype(np.float64) * depth_scale
    roi_depth = depth_m[y_min:y_max, x_min:x_max]
    valid = (roi_depth > depth_cfg["min_m"]) & (roi_depth < depth_cfg["max_m"])
    if color_cfg and color_cfg.get("enabled", False):
        hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[y_min:y_max, x_min:x_max, 1]
        val = hsv[y_min:y_max, x_min:x_max, 2]
        valid &= (sat <= int(color_cfg["saturation_max"])) & (val >= int(color_cfg["value_min"]))
    ys_roi, xs_roi = np.nonzero(valid)
    us = xs_roi + x_min
    vs = ys_roi + y_min
    z = depth_m[vs, us]
    x = (us - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vs - intrinsics["ppy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z))
    colors = color_rgb[vs, us].astype(np.float64) / 255.0
    pixels = np.column_stack((us, vs))
    return points, colors, pixels, (x_min, y_min, x_max, y_max)


def cluster_score(extent, cluster_cfg):
    target = np.array([0.05645, 0.01469, 0.01185])
    dims = np.sort(np.asarray(extent, dtype=np.float64))[::-1]
    relative = np.abs(dims - target) / target
    min_long = float(cluster_cfg.get("min_long_extent_m", 0.0))
    max_long = float(cluster_cfg.get("max_long_extent_m", 999.0))
    long_penalty = 0.0
    if dims[0] < min_long:
        long_penalty += (min_long - dims[0]) * 80.0
    if dims[0] > max_long:
        long_penalty += (dims[0] - max_long) * 80.0
    # Prioritize recovering the plug length; partial short clusters are not useful initializations.
    return float(relative[0] * 4.0 + relative[1] + relative[2] + long_penalty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--roi_key", default="cluster_roi_tagged")
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--select_labels",
        help="Comma-separated DBSCAN labels to export as the selected cluster, e.g. 6 or 5,6.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    markerless = config["markerless"]
    roi = markerless[args.roi_key]
    depth_cfg = markerless["depth"]
    cluster_cfg = markerless["clustering"]
    color_cfg = cluster_cfg.get("color_filter", {})

    pipeline, profile = start_playback(args.bag)
    try:
        from plug_pose.bag_reader import get_color_intrinsics

        intrinsics = get_color_intrinsics(profile)
        depth_scale = get_depth_scale(profile)
    finally:
        pipeline.stop()
    if depth_scale is None:
        raise RuntimeError("No depth scale found in bag.")

    frame = load_frame(args.bag, args.frame_id)
    points, colors, pixels, roi_box = backproject_with_pixels(
        frame.depth_z16,
        frame.color_rgb,
        intrinsics,
        depth_scale,
        roi,
        depth_cfg,
        color_cfg,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(cluster_cfg["dbscan_eps_m"]),
            min_points=int(cluster_cfg["dbscan_min_points"]),
            print_progress=False,
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    roi_image = color_bgr.copy()
    x_min, y_min, x_max, y_max = roi_box
    cv2.rectangle(roi_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_roi.png"), roi_image)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_depth.png"), colorize_depth(frame.depth_z16))

    overlay = color_bgr.copy()
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_points = []
    cluster_colors = []
    rows = []

    best_label = None
    best_score = float("inf")
    min_cluster_points = int(cluster_cfg["min_cluster_points"])
    label_centers_px = {}
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if len(idx) < min_cluster_points:
            continue
        cluster = pcd.select_by_index(idx.tolist())
        aabb = cluster.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        center = aabb.get_center()
        score = cluster_score(extent, cluster_cfg)
        if score < best_score:
            best_score = score
            best_label = label

        rows.append(
            {
                "label": label,
                "points": len(idx),
                "center_x": f"{center[0]:.6f}",
                "center_y": f"{center[1]:.6f}",
                "center_z": f"{center[2]:.6f}",
                "extent_x": f"{extent[0]:.6f}",
                "extent_y": f"{extent[1]:.6f}",
                "extent_z": f"{extent[2]:.6f}",
                "extent_sorted_desc": " ".join(f"{v:.6f}" for v in np.sort(extent)[::-1]),
                "plug_score": f"{score:.6f}",
            }
        )

        rgb = PALETTE[label % len(PALETTE)]
        cluster_points.append(points[idx])
        cluster_colors.append(np.tile(rgb.astype(np.float64) / 255.0, (len(idx), 1)))
        label_centers_px[label] = np.round(np.mean(pixels[idx], axis=0)).astype(int)
        for u, v in pixels[idx]:
            overlay[v, u] = (0.45 * overlay[v, u] + 0.55 * rgb[::-1]).astype(np.uint8)

        o3d.io.write_point_cloud(str(args.out_dir / f"frame_{frame.frame_id:06d}_cluster_{label:03d}.ply"), cluster)

    if cluster_points:
        cluster_pcd.points = o3d.utility.Vector3dVector(np.vstack(cluster_points))
        cluster_pcd.colors = o3d.utility.Vector3dVector(np.vstack(cluster_colors))
        o3d.io.write_point_cloud(str(args.out_dir / f"frame_{frame.frame_id:06d}_clusters_colored.ply"), cluster_pcd)

    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    for label, center_px in label_centers_px.items():
        u, v = int(center_px[0]), int(center_px[1])
        cv2.putText(overlay, str(label), (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, str(label), (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_clusters_rgb.png"), overlay)

    selected_labels = None
    if args.select_labels:
        selected_labels = [int(label.strip()) for label in args.select_labels.split(",") if label.strip()]
    elif best_label is not None:
        selected_labels = [int(best_label)]

    if selected_labels:
        selected_idx = np.where(np.isin(labels, selected_labels))[0]
        best_mask = np.zeros(frame.depth_z16.shape, dtype=np.uint8)
        best_mask[pixels[selected_idx, 1], pixels[selected_idx, 0]] = 255
        best_overlay = color_bgr.copy()
        green = np.zeros_like(best_overlay)
        green[:, :, 1] = 255
        best_overlay = np.where(best_mask[:, :, None] > 0, cv2.addWeighted(best_overlay, 0.4, green, 0.6, 0), best_overlay)
        cv2.rectangle(best_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        label_text = ",".join(str(label) for label in selected_labels)
        cv2.putText(best_overlay, f"selected labels: {label_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(best_overlay, f"selected labels: {label_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_selected_cluster_rgb.png"), best_overlay)
        cv2.imwrite(str(args.out_dir / f"frame_{frame.frame_id:06d}_selected_cluster_mask.png"), best_mask)

        selected_pcd = pcd.select_by_index(selected_idx.tolist())
        o3d.io.write_point_cloud(str(args.out_dir / f"frame_{frame.frame_id:06d}_selected_cluster.ply"), selected_pcd)

    with (args.out_dir / f"frame_{frame.frame_id:06d}_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "label",
            "points",
            "center_x",
            "center_y",
            "center_z",
            "extent_x",
            "extent_y",
            "extent_z",
            "extent_sorted_desc",
            "plug_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["plug_score"])))

    print(f"frame_id: {frame.frame_id}")
    print(f"valid ROI depth points: {len(points)}")
    if color_cfg.get("enabled", False):
        print(
            "color filter: "
            f"saturation <= {color_cfg['saturation_max']}, value >= {color_cfg['value_min']}"
        )
    print(f"clusters kept: {len(rows)}")
    print(f"best plug-like cluster label: {best_label}")
    if selected_labels:
        print(f"selected labels exported: {selected_labels}")
    print(f"saved cluster debug outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
