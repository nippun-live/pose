from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.transforms import pose_quat_xyzw_to_matrix  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if row.get("detected", "1") != "1":
                continue
            if not row.get("tx"):
                continue
            rows[int(row["frame_id"])] = row
        return rows


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def rotation_error_deg(gt: np.ndarray, pred: np.ndarray) -> float:
    delta = gt[:3, :3].T @ pred[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.parent.name or path.stem, path


def summarize_run(name: str, pred_path: Path, gt_rows: dict[int, dict[str, str]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    pred_rows = load_pose_csv(pred_path)
    matched_frames = sorted(set(gt_rows) & set(pred_rows))

    per_frame = []
    trans_errors_m = []
    rot_errors_deg = []
    fitness_values = []
    rmse_values_m = []

    for frame_id in matched_frames:
        gt_t = row_to_transform(gt_rows[frame_id])
        pred_t = row_to_transform(pred_rows[frame_id])
        trans_error = float(np.linalg.norm(pred_t[:3, 3] - gt_t[:3, 3]))
        rot_error = rotation_error_deg(gt_t, pred_t)
        trans_errors_m.append(trans_error)
        rot_errors_deg.append(rot_error)

        row = pred_rows[frame_id]
        if row.get("fitness"):
            fitness_values.append(float(row["fitness"]))
        if row.get("rmse"):
            rmse_values_m.append(float(row["rmse"]))

        per_frame.append(
            {
                "run": name,
                "frame_id": frame_id,
                "translation_error_m": f"{trans_error:.9f}",
                "translation_error_mm": f"{trans_error * 1000.0:.3f}",
                "rotation_error_deg": f"{rot_error:.6f}",
                "fitness": row.get("fitness", ""),
                "rmse_m": row.get("rmse", ""),
            }
        )

    def stat(values: list[float], fn, scale: float = 1.0) -> str:
        if not values:
            return ""
        return f"{fn(values) * scale:.6f}"

    summary = {
        "run": name,
        "pred_csv": str(pred_path),
        "gt_frames": str(len(gt_rows)),
        "pred_frames": str(len(pred_rows)),
        "matched_frames": str(len(matched_frames)),
        "mean_translation_error_mm": stat(trans_errors_m, statistics.mean, 1000.0),
        "median_translation_error_mm": stat(trans_errors_m, statistics.median, 1000.0),
        "p90_translation_error_mm": f"{percentile(trans_errors_m, 90) * 1000.0:.6f}" if trans_errors_m else "",
        "max_translation_error_mm": stat(trans_errors_m, max, 1000.0),
        "mean_rotation_error_deg": stat(rot_errors_deg, statistics.mean),
        "median_rotation_error_deg": stat(rot_errors_deg, statistics.median),
        "p90_rotation_error_deg": f"{percentile(rot_errors_deg, 90):.6f}" if rot_errors_deg else "",
        "max_rotation_error_deg": stat(rot_errors_deg, max),
        "mean_fitness": stat(fitness_values, statistics.mean),
        "mean_rmse_mm": stat(rmse_values_m, statistics.mean, 1000.0),
    }
    return summary, per_frame


def print_markdown_table(summaries: list[dict[str, str]]) -> None:
    columns = [
        "run",
        "matched_frames",
        "mean_translation_error_mm",
        "median_translation_error_mm",
        "mean_rotation_error_deg",
        "median_rotation_error_deg",
        "mean_fitness",
        "mean_rmse_mm",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in summaries:
        print("| " + " | ".join(row.get(column, "") for column in columns) + " |")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare markerless plug pose CSVs against ArUco-derived plug pose.")
    parser.add_argument("--gt_csv", type=Path, required=True, help="ArUco-derived plug pose CSV.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction CSV, optionally named as name=path/to/poses.csv. Can be repeated.",
    )
    parser.add_argument("--out_summary_csv", type=Path)
    parser.add_argument("--out_per_frame_csv", type=Path)
    args = parser.parse_args()

    gt_rows = load_pose_csv(args.gt_csv)
    summaries = []
    all_per_frame = []
    for pred_arg in args.pred:
        name, path = parse_prediction_arg(pred_arg)
        summary, per_frame = summarize_run(name, path, gt_rows)
        summaries.append(summary)
        all_per_frame.extend(per_frame)

    print_markdown_table(summaries)
    if args.out_summary_csv:
        write_csv(args.out_summary_csv, summaries)
    if args.out_per_frame_csv:
        write_csv(args.out_per_frame_csv, all_per_frame)


if __name__ == "__main__":
    main()
