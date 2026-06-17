from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.transforms import matrix_to_quaternion_xyzw, pose_quat_xyzw_to_matrix  # noqa: E402


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def detected_pose_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {
        int(row["frame_id"]): row
        for row in rows
        if row.get("detected", "1") == "1" and row.get("tx") and row.get("qx")
    }


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def rotation_error_deg(gt: np.ndarray, pred: np.ndarray) -> float:
    delta = gt[:3, :3].T @ pred[:3, :3]
    angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def parse_ranges(values: list[str] | None, frames: list[int]) -> list[tuple[str, list[int]]]:
    if not values:
        return [("all", frames)]
    ranges = []
    for value in values:
        if ":" in value:
            label, span = value.split(":", 1)
        else:
            label = value
            span = value
        start_s, end_s = span.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        ranges.append((label, [frame for frame in frames if start <= frame <= end]))
    return ranges


def fit_local_correction(
    gt_by_frame: dict[int, dict[str, str]],
    pred_by_frame: dict[int, dict[str, str]],
    train_frames: list[int],
) -> np.ndarray:
    rotations = []
    local_translations = []
    for frame_id in train_frames:
        gt_t = row_to_transform(gt_by_frame[frame_id])
        pred_t = row_to_transform(pred_by_frame[frame_id])
        local_delta = np.linalg.inv(pred_t) @ gt_t
        rotations.append(Rotation.from_matrix(local_delta[:3, :3]))
        local_translations.append(local_delta[:3, 3])
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = Rotation.concatenate(rotations).mean().as_matrix()
    correction[:3, 3] = np.mean(local_translations, axis=0)
    return correction


def apply_correction_to_rows(rows: list[dict[str, str]], correction: np.ndarray) -> list[dict[str, str]]:
    corrected = []
    for row in rows:
        output = dict(row)
        if row.get("detected", "1") == "1" and row.get("tx") and row.get("qx"):
            transform = row_to_transform(row) @ correction
            quat = matrix_to_quaternion_xyzw(transform)
            output["tx"] = f"{transform[0, 3]:.9f}"
            output["ty"] = f"{transform[1, 3]:.9f}"
            output["tz"] = f"{transform[2, 3]:.9f}"
            output["qx"] = f"{quat[0]:.9f}"
            output["qy"] = f"{quat[1]:.9f}"
            output["qz"] = f"{quat[2]:.9f}"
            output["qw"] = f"{quat[3]:.9f}"
        corrected.append(output)
    return corrected


def evaluate_frames(
    gt_by_frame: dict[int, dict[str, str]],
    pred_by_frame: dict[int, dict[str, str]],
    frames: list[int],
) -> dict[str, str]:
    trans_errors = []
    rot_errors = []
    for frame_id in frames:
        gt_t = row_to_transform(gt_by_frame[frame_id])
        pred_t = row_to_transform(pred_by_frame[frame_id])
        trans_errors.append(float(np.linalg.norm(pred_t[:3, 3] - gt_t[:3, 3]) * 1000.0))
        rot_errors.append(rotation_error_deg(gt_t, pred_t))

    def stats(values: list[float]) -> tuple[float, float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return (
            float(np.mean(values)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
            float(np.max(values)),
        )

    t_mean, t_median, t_p90, t_max = stats(trans_errors)
    r_mean, r_median, r_p90, r_max = stats(rot_errors)
    return {
        "n": str(len(frames)),
        "mean_translation_error_mm": f"{t_mean:.6f}",
        "median_translation_error_mm": f"{t_median:.6f}",
        "p90_translation_error_mm": f"{t_p90:.6f}",
        "max_translation_error_mm": f"{t_max:.6f}",
        "mean_rotation_error_deg": f"{r_mean:.6f}",
        "median_rotation_error_deg": f"{r_median:.6f}",
        "p90_rotation_error_deg": f"{r_p90:.6f}",
        "max_rotation_error_deg": f"{r_max:.6f}",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and apply a fixed local correction from markerless poses to ArUco poses.")
    parser.add_argument("--gt_csv", type=Path, required=True)
    parser.add_argument("--pred_csv", type=Path, required=True)
    parser.add_argument("--train_start", type=int)
    parser.add_argument("--train_end", type=int)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_summary_csv", type=Path)
    parser.add_argument(
        "--eval_range",
        action="append",
        help="Optional named range label:start-end or start-end. Can be repeated.",
    )
    args = parser.parse_args()

    gt_rows = load_csv_rows(args.gt_csv)
    pred_rows = load_csv_rows(args.pred_csv)
    gt_by_frame = detected_pose_rows(gt_rows)
    pred_by_frame = detected_pose_rows(pred_rows)
    matched_frames = sorted(set(gt_by_frame) & set(pred_by_frame))
    if not matched_frames:
        raise RuntimeError("No matched detected frames between GT and prediction CSVs.")

    train_frames = matched_frames
    if args.train_start is not None or args.train_end is not None:
        start = args.train_start if args.train_start is not None else matched_frames[0]
        end = args.train_end if args.train_end is not None else matched_frames[-1]
        train_frames = [frame for frame in matched_frames if start <= frame <= end]
    if not train_frames:
        raise RuntimeError("No training frames selected for correction calibration.")

    correction = fit_local_correction(gt_by_frame, pred_by_frame, train_frames)
    corrected_rows = apply_correction_to_rows(pred_rows, correction)
    write_rows(args.out_csv, corrected_rows)

    corrected_by_frame = detected_pose_rows(corrected_rows)
    ranges = parse_ranges(args.eval_range, matched_frames)
    summaries = []
    for label, frames in ranges:
        raw = evaluate_frames(gt_by_frame, pred_by_frame, frames)
        corrected = evaluate_frames(gt_by_frame, corrected_by_frame, frames)
        for kind, metrics in [("raw", raw), ("corrected", corrected)]:
            row = {
                "range": label,
                "kind": kind,
                "train_start": str(train_frames[0]),
                "train_end": str(train_frames[-1]),
                "correction_euler_xyz_deg": " ".join(f"{v:.6f}" for v in Rotation.from_matrix(correction[:3, :3]).as_euler("xyz", degrees=True)),
                "correction_translation_m": " ".join(f"{v:.9f}" for v in correction[:3, 3]),
            }
            row.update(metrics)
            summaries.append(row)

    print("correction_euler_xyz_deg:", summaries[0]["correction_euler_xyz_deg"])
    print("correction_translation_m:", summaries[0]["correction_translation_m"])
    for row in summaries:
        print(
            row["range"],
            row["kind"],
            "n=" + row["n"],
            "t_mean_mm=" + row["mean_translation_error_mm"],
            "r_mean_deg=" + row["mean_rotation_error_deg"],
        )
    if args.out_summary_csv:
        write_rows(args.out_summary_csv, summaries)


if __name__ == "__main__":
    main()
