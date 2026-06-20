from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plug_pose.transforms import pose_quat_xyzw_to_matrix  # noqa: E402


def load_pose_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if row.get("detected", "1") == "1" and row.get("tx") and row.get("qx"):
                rows[int(row["frame_id"])] = row
        return rows


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def parse_pred(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.parent.name or path.stem, path


def correction_deltas(
    gt_rows: dict[int, dict[str, str]],
    pred_rows: dict[int, dict[str, str]],
    train_start: int | None,
    train_end: int | None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    frames = sorted(set(gt_rows) & set(pred_rows))
    train_frames = [
        frame
        for frame in frames
        if (train_start is None or frame >= train_start) and (train_end is None or frame <= train_end)
    ]
    if not train_frames:
        raise RuntimeError("No matched training frames.")

    rows = []
    rotations = []
    translations = []
    for frame_id in frames:
        gt_t = row_to_transform(gt_rows[frame_id])
        pred_t = row_to_transform(pred_rows[frame_id])
        local_delta = np.linalg.inv(pred_t) @ gt_t
        rot = Rotation.from_matrix(local_delta[:3, :3])
        rows.append(
            {
                "frame_id": frame_id,
                "correction": local_delta,
                "rotation": rot,
                "translation": local_delta[:3, 3].copy(),
                "in_train": frame_id in train_frames,
            }
        )
        if frame_id in train_frames:
            rotations.append(rot)
            translations.append(local_delta[:3, 3].copy())

    mean_correction = np.eye(4, dtype=np.float64)
    mean_correction[:3, :3] = Rotation.concatenate(rotations).mean().as_matrix()
    mean_correction[:3, 3] = np.mean(translations, axis=0)
    return rows, mean_correction


def summarize(name: str, rows: list[dict[str, object]], mean_correction: np.ndarray) -> tuple[dict[str, str], list[dict[str, str]]]:
    mean_rot = Rotation.from_matrix(mean_correction[:3, :3])
    mean_translation = mean_correction[:3, 3]

    per_frame = []
    trans_dev_mm = []
    rot_dev_deg = []
    tx_mm = []
    ty_mm = []
    tz_mm = []
    for row in rows:
        translation = row["translation"]
        rotation = row["rotation"]
        assert isinstance(translation, np.ndarray)
        assert isinstance(rotation, Rotation)
        trans_dev = float(np.linalg.norm(translation - mean_translation) * 1000.0)
        rot_dev = float((mean_rot.inv() * rotation).magnitude() * 180.0 / np.pi)
        trans_dev_mm.append(trans_dev)
        rot_dev_deg.append(rot_dev)
        tx_mm.append(float(translation[0] * 1000.0))
        ty_mm.append(float(translation[1] * 1000.0))
        tz_mm.append(float(translation[2] * 1000.0))
        euler = rotation.as_euler("xyz", degrees=True)
        per_frame.append(
            {
                "run": name,
                "frame_id": str(row["frame_id"]),
                "in_train": str(int(bool(row["in_train"]))),
                "correction_tx_mm": f"{translation[0] * 1000.0:.6f}",
                "correction_ty_mm": f"{translation[1] * 1000.0:.6f}",
                "correction_tz_mm": f"{translation[2] * 1000.0:.6f}",
                "correction_rx_deg": f"{euler[0]:.6f}",
                "correction_ry_deg": f"{euler[1]:.6f}",
                "correction_rz_deg": f"{euler[2]:.6f}",
                "translation_deviation_from_mean_mm": f"{trans_dev:.6f}",
                "rotation_deviation_from_mean_deg": f"{rot_dev:.6f}",
            }
        )

    def stat(values: list[float], fn) -> str:
        return f"{fn(values):.6f}" if values else ""

    summary = {
        "run": name,
        "frames": str(len(rows)),
        "mean_correction_tx_mm": f"{mean_translation[0] * 1000.0:.6f}",
        "mean_correction_ty_mm": f"{mean_translation[1] * 1000.0:.6f}",
        "mean_correction_tz_mm": f"{mean_translation[2] * 1000.0:.6f}",
        "std_correction_tx_mm": stat(tx_mm, statistics.stdev) if len(tx_mm) > 1 else "0.000000",
        "std_correction_ty_mm": stat(ty_mm, statistics.stdev) if len(ty_mm) > 1 else "0.000000",
        "std_correction_tz_mm": stat(tz_mm, statistics.stdev) if len(tz_mm) > 1 else "0.000000",
        "mean_translation_deviation_mm": stat(trans_dev_mm, statistics.mean),
        "median_translation_deviation_mm": stat(trans_dev_mm, statistics.median),
        "p90_translation_deviation_mm": f"{np.percentile(trans_dev_mm, 90):.6f}" if trans_dev_mm else "",
        "max_translation_deviation_mm": stat(trans_dev_mm, max),
        "mean_rotation_deviation_deg": stat(rot_dev_deg, statistics.mean),
        "median_rotation_deviation_deg": stat(rot_dev_deg, statistics.median),
        "p90_rotation_deviation_deg": f"{np.percentile(rot_dev_deg, 90):.6f}" if rot_dev_deg else "",
        "max_rotation_deviation_deg": stat(rot_dev_deg, max),
    }
    return summary, per_frame


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(per_frame: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_run: dict[str, list[dict[str, str]]] = {}
    for row in per_frame:
        by_run.setdefault(row["run"], []).append(row)
    for rows in by_run.values():
        rows.sort(key=lambda row: int(row["frame_id"]))

    plots = [
        ("translation_deviation_from_mean_mm", "Correction Translation Deviation from Fixed Mean (mm)", "correction_translation_deviation.png"),
        ("rotation_deviation_from_mean_deg", "Correction Rotation Deviation from Fixed Mean (deg)", "correction_rotation_deviation.png"),
    ]
    for key, ylabel, filename in plots:
        plt.figure(figsize=(13, 7))
        for run, rows in by_run.items():
            plt.plot(
                [int(row["frame_id"]) for row in rows],
                [float(row[key]) for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=run,
            )
        plt.xlabel("Frame number")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()

    component_keys = ["correction_tx_mm", "correction_ty_mm", "correction_tz_mm"]
    for run, rows in by_run.items():
        plt.figure(figsize=(13, 7))
        for key in component_keys:
            plt.plot(
                [int(row["frame_id"]) for row in rows],
                [float(row[key]) for row in rows],
                marker="o",
                markersize=2.2,
                linewidth=1.1,
                label=key.replace("correction_", "").replace("_mm", ""),
            )
        plt.xlabel("Frame number")
        plt.ylabel("Per-frame correction translation component (mm)")
        plt.title(run)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{run}_translation_components.png", dpi=180)
        plt.close()


def print_table(summaries: list[dict[str, str]]) -> None:
    columns = [
        "run",
        "frames",
        "mean_translation_deviation_mm",
        "median_translation_deviation_mm",
        "p90_translation_deviation_mm",
        "mean_rotation_deviation_deg",
        "median_rotation_deviation_deg",
        "p90_rotation_deviation_deg",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in summaries:
        print("| " + " | ".join(row[column] for column in columns) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether a fixed pose-frame correction is actually stable over frames.")
    parser.add_argument("--gt_csv", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path/to/raw_prediction.csv")
    parser.add_argument("--train_start", type=int, default=127)
    parser.add_argument("--train_end", type=int, default=170)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/phase3/correction_variance"))
    args = parser.parse_args()

    gt_rows = load_pose_csv(args.gt_csv)
    summaries = []
    all_per_frame = []
    for pred_arg in args.pred:
        name, pred_path = parse_pred(pred_arg)
        pred_rows = load_pose_csv(pred_path)
        rows, mean_correction = correction_deltas(gt_rows, pred_rows, args.train_start, args.train_end)
        summary, per_frame = summarize(name, rows, mean_correction)
        summaries.append(summary)
        all_per_frame.extend(per_frame)

    print_table(summaries)
    write_csv(args.out_dir / "correction_variance_summary.csv", summaries)
    write_csv(args.out_dir / "correction_variance_per_frame.csv", all_per_frame)
    plot_runs(all_per_frame, args.out_dir / "plots")
    print(f"saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
