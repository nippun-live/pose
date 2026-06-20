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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_correction(summary_csv: Path) -> np.ndarray:
    rows = load_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"No rows in {summary_csv}")
    row = next((item for item in rows if item.get("kind") == "corrected"), rows[0])
    euler = [float(value) for value in row["correction_euler_xyz_deg"].split()]
    trans = [float(value) for value in row["correction_translation_m"].split()]
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    correction[:3, 3] = np.array(trans, dtype=np.float64)
    return correction


def row_to_transform(row: dict[str, str]) -> np.ndarray:
    return pose_quat_xyzw_to_matrix(
        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
    )


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a fixed local pose-frame correction to a pose CSV.")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--correction_summary_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    correction = load_correction(args.correction_summary_csv)
    rows = load_rows(args.poses)
    corrected = []
    updated = 0
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
            updated += 1
        corrected.append(output)

    write_rows(args.out_csv, corrected)
    print("correction_euler_xyz_deg:", " ".join(f"{v:.6f}" for v in Rotation.from_matrix(correction[:3, :3]).as_euler("xyz", degrees=True)))
    print("correction_translation_m:", " ".join(f"{v:.9f}" for v in correction[:3, 3]))
    print(f"updated {updated} pose rows")
    print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
