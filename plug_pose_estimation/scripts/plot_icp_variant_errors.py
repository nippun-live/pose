from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DISPLAY_NAMES = {
    "sam2_p2p": "point-to-point",
    "p2plane": "point-to-plane",
    "robust_p2plane": "robust point-to-plane",
    "ms_p2p": "multiscale point-to-point",
    "ms_p2plane": "multiscale point-to-plane",
    "ms_robust_p2plane": "multiscale robust point-to-plane",
}


def load_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    runs: dict[str, list[dict[str, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run = row["run"]
            runs.setdefault(run, []).append(
                {
                    "frame_id": int(row["frame_id"]),
                    "translation_error_mm": float(row["translation_error_mm"]),
                    "rotation_error_deg": float(row["rotation_error_deg"]),
                }
            )
    for rows in runs.values():
        rows.sort(key=lambda item: item["frame_id"])
    return runs


def plot_metric(
    runs: dict[str, list[dict[str, float]]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(13, 7))
    for run, rows in runs.items():
        x = [row["frame_id"] for row in rows]
        y = [row[metric] for row in rows]
        plt.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=DISPLAY_NAMES.get(run, run),
        )

    plt.title(title)
    plt.xlabel("Frame number")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-frame ICP errors against ArUco reference.")
    parser.add_argument(
        "--eval_csv",
        type=Path,
        default=Path("outputs/phase3/sam2_icp_variant_eval_per_frame.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/phase3/plots"),
    )
    args = parser.parse_args()

    runs = load_rows(args.eval_csv)
    plot_metric(
        runs,
        "translation_error_mm",
        "Translation error vs ArUco plug pose (mm)",
        "ICP Variant Translation Error by Frame",
        args.out_dir / "icp_variant_translation_error.png",
    )
    plot_metric(
        runs,
        "rotation_error_deg",
        "Rotation error vs ArUco plug pose (degrees)",
        "ICP Variant Rotation Error by Frame",
        args.out_dir / "icp_variant_rotation_error.png",
    )
    print(f"saved {args.out_dir / 'icp_variant_translation_error.png'}")
    print(f"saved {args.out_dir / 'icp_variant_rotation_error.png'}")


if __name__ == "__main__":
    main()
