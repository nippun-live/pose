from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_pose_correction import (  # noqa: E402
    apply_correction_to_rows,
    detected_pose_rows,
    evaluate_frames,
    fit_local_correction,
    load_csv_rows,
    write_rows,
)


def write_frame_list(path: Path, frames: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id"])
        writer.writeheader()
        for frame_id in frames:
            writer.writerow({"frame_id": frame_id})


def summarize_split(
    correction: np.ndarray,
    gt_by_frame: dict[int, dict[str, str]],
    pred_by_frame: dict[int, dict[str, str]],
    corrected_by_frame: dict[int, dict[str, str]],
    split_frames: list[tuple[str, list[int]]],
    train_frames: list[int],
) -> list[dict[str, str]]:
    summaries = []
    correction_euler = " ".join(
        f"{value:.6f}"
        for value in Rotation.from_matrix(correction[:3, :3]).as_euler("xyz", degrees=True)
    )
    correction_translation = " ".join(f"{value:.9f}" for value in correction[:3, 3])
    for label, frames in split_frames:
        raw = evaluate_frames(gt_by_frame, pred_by_frame, frames)
        corrected = evaluate_frames(gt_by_frame, corrected_by_frame, frames)
        for kind, metrics in [("raw", raw), ("corrected", corrected)]:
            row = {
                "split": label,
                "kind": kind,
                "train_count": str(len(train_frames)),
                "train_first_frame": str(train_frames[0]),
                "train_last_frame": str(train_frames[-1]),
                "correction_euler_xyz_deg": correction_euler,
                "correction_translation_m": correction_translation,
            }
            row.update(metrics)
            summaries.append(row)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a fixed markerless-to-ArUco correction on a split of ArUco-visible frames."
    )
    parser.add_argument("--gt_csv", type=Path, required=True)
    parser.add_argument("--pred_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_summary_csv", type=Path, required=True)
    parser.add_argument("--out_train_frames_csv", type=Path)
    parser.add_argument("--out_validation_frames_csv", type=Path)
    parser.add_argument(
        "--split_mode",
        choices=["alternate", "random"],
        default="alternate",
        help="How to split matched ArUco-visible frames into train and validation sets.",
    )
    parser.add_argument(
        "--train_offset",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 trains on the first, third, fifth... matched ArUco-visible frames. 1 uses the opposite half.",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.6,
        help="Training fraction for --split_mode random. Sampling is without replacement.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --split_mode random.")
    args = parser.parse_args()

    gt_rows = load_csv_rows(args.gt_csv)
    pred_rows = load_csv_rows(args.pred_csv)
    gt_by_frame = detected_pose_rows(gt_rows)
    pred_by_frame = detected_pose_rows(pred_rows)
    matched_frames = sorted(set(gt_by_frame) & set(pred_by_frame))
    if not matched_frames:
        raise RuntimeError("No matched detected frames between GT and prediction CSVs.")

    if args.split_mode == "alternate":
        train_frames = matched_frames[args.train_offset :: 2]
        validation_frames = matched_frames[1 - args.train_offset :: 2]
    else:
        if not 0.0 < args.train_fraction < 1.0:
            raise RuntimeError("--train_fraction must be between 0 and 1 for random split.")
        rng = random.Random(args.seed)
        shuffled = list(matched_frames)
        rng.shuffle(shuffled)
        train_count = int(round(len(shuffled) * args.train_fraction))
        train_count = min(max(train_count, 1), len(shuffled) - 1)
        train_frames = sorted(shuffled[:train_count])
        validation_frames = sorted(shuffled[train_count:])
    if not train_frames or not validation_frames:
        raise RuntimeError("Split did not produce both train and validation frames.")

    correction = fit_local_correction(gt_by_frame, pred_by_frame, train_frames)
    corrected_rows = apply_correction_to_rows(pred_rows, correction)
    write_rows(args.out_csv, corrected_rows)
    corrected_by_frame = detected_pose_rows(corrected_rows)

    early = [frame for frame in matched_frames if 0 <= frame <= 126]
    middle = [frame for frame in matched_frames if 127 <= frame <= 170]
    late = [frame for frame in matched_frames if 171 <= frame <= 354]
    split_frames = [
        ("alternate_train", train_frames),
        ("alternate_validation", validation_frames),
        ("all_matched", matched_frames),
        ("early_block_0_126", early),
        ("middle_block_127_170", middle),
        ("late_block_171_354", late),
    ]
    summaries = summarize_split(correction, gt_by_frame, pred_by_frame, corrected_by_frame, split_frames, train_frames)
    for row in summaries:
        row["split_mode"] = args.split_mode
        row["train_fraction"] = f"{args.train_fraction:.6f}" if args.split_mode == "random" else ""
        row["seed"] = str(args.seed) if args.split_mode == "random" else ""
    write_rows(args.out_summary_csv, summaries)

    if args.out_train_frames_csv:
        write_frame_list(args.out_train_frames_csv, train_frames)
    if args.out_validation_frames_csv:
        write_frame_list(args.out_validation_frames_csv, validation_frames)

    print("matched_frames:", len(matched_frames))
    print("split_mode:", args.split_mode)
    if args.split_mode == "random":
        print("train_fraction:", args.train_fraction)
        print("seed:", args.seed)
    print("train_frames:", len(train_frames), f"{train_frames[0]}-{train_frames[-1]}")
    print("validation_frames:", len(validation_frames), f"{validation_frames[0]}-{validation_frames[-1]}")
    print("correction_euler_xyz_deg:", summaries[0]["correction_euler_xyz_deg"])
    print("correction_translation_m:", summaries[0]["correction_translation_m"])
    for row in summaries:
        print(
            row["split"],
            row["kind"],
            "n=" + row["n"],
            "t_mean_mm=" + row["mean_translation_error_mm"],
            "r_mean_deg=" + row["mean_rotation_error_deg"],
        )


if __name__ == "__main__":
    main()
