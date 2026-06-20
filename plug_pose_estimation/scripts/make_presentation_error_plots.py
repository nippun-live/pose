from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    out_dir = Path("presentation_assets")
    out_dir.mkdir(exist_ok=True)

    per_frame = pd.read_csv("outputs/phase4/run1/evaluation_per_frame_with_random60_correction.csv")
    raw = per_frame[per_frame["run"] == "multiscale_p2p_raw"]
    corrected = per_frame[per_frame["run"] == "multiscale_p2p_corr_random60"]

    make_plot(
        raw,
        corrected,
        raw_y=raw["translation_error_mm"],
        corrected_y=corrected["translation_error_mm"],
        title="Multiscale P2P Translation Error Before/After Fixed Correction",
        ylabel="Error vs ArUco (mm)",
        out_path=out_dir / "multiscale_p2p_raw_vs_corrected_translation_error.png",
    )

    make_plot(
        raw,
        corrected,
        raw_y=raw["rotation_error_deg"],
        corrected_y=corrected["rotation_error_deg"],
        title="Multiscale P2P Rotation Error Before/After Fixed Correction",
        ylabel="Error vs ArUco (deg)",
        out_path=out_dir / "multiscale_p2p_raw_vs_corrected_rotation_error.png",
    )


def make_plot(
    raw: pd.DataFrame,
    corrected: pd.DataFrame,
    raw_y: pd.Series,
    corrected_y: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    ax.plot(raw["frame_id"], raw_y, label="raw multiscale P2P", color="#d97706", linewidth=1.4)
    ax.plot(
        corrected["frame_id"],
        corrected_y,
        label="corrected random60",
        color="#059669",
        linewidth=1.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
