from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def add_sam2_to_path(sam2_repo: Path) -> None:
    repo = sam2_repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    sys.path.insert(0, str(repo))


def load_frame_map(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_prompt(value: str) -> tuple[int, float, float, int]:
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Prompt must be FRAME:X:Y or FRAME:X:Y:LABEL, got {value!r}")
    frame = int(parts[0])
    x = float(parts[1])
    y = float(parts[2])
    label = int(parts[3]) if len(parts) == 4 else 1
    if label not in (0, 1):
        raise ValueError(f"Prompt label must be 0 or 1, got {label}")
    return frame, x, y, label


def prompts_to_sam_frames(
    prompt_values: list[str],
    bag_to_sam: dict[int, int],
    prompts_are_sam_idx: bool,
) -> dict[int, list[tuple[float, float, int]]]:
    prompts_by_frame: dict[int, list[tuple[float, float, int]]] = {}
    for value in prompt_values:
        frame, x, y, label = parse_prompt(value)
        sam_idx = frame if prompts_are_sam_idx else bag_to_sam[frame]
        prompts_by_frame.setdefault(int(sam_idx), []).append((x, y, label))
    return prompts_by_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=Path, required=True, help="Directory containing SAM2 JPEG frames.")
    parser.add_argument("--frame_map", type=Path, required=True)
    parser.add_argument("--sam2_repo", type=Path, default=Path("third_party/sam2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("third_party/sam2/checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--click_frame", type=int, required=True, help="Original bag frame id or SAM frame idx.")
    parser.add_argument("--click_is_sam_idx", action="store_true")
    parser.add_argument("--click_x", type=float, required=True)
    parser.add_argument("--click_y", type=float, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Additional corrective prompt as FRAME:X:Y:LABEL. LABEL 1=plug, 0=not plug. Can be repeated.",
    )
    parser.add_argument("--prompt_is_sam_idx", action="store_true")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--reverse", action="store_true", help="Propagate masks backward through the video.")
    parser.add_argument("--propagate_start_frame", type=int, help="Original bag frame id or SAM frame idx to start propagation.")
    parser.add_argument("--propagate_start_is_sam_idx", action="store_true")
    parser.add_argument("--max_track_frames", type=int)
    args = parser.parse_args()

    add_sam2_to_path(args.sam2_repo)
    from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415

    frame_map = load_frame_map(args.frame_map)
    bag_to_sam = {int(row["bag_frame_id"]): int(row["sam_frame_idx"]) for row in frame_map}
    click_sam_idx = int(args.click_frame) if args.click_is_sam_idx else bag_to_sam[int(args.click_frame)]
    propagate_start_idx = None
    if args.propagate_start_frame is not None:
        propagate_start_idx = (
            int(args.propagate_start_frame)
            if args.propagate_start_is_sam_idx
            else bag_to_sam[int(args.propagate_start_frame)]
        )
    prompts_by_frame = prompts_to_sam_frames(args.prompt, bag_to_sam, args.prompt_is_sam_idx)
    prompts_by_frame.setdefault(click_sam_idx, []).insert(0, (args.click_x, args.click_y, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "masks"
    overlay_dir = args.out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: PLC0415

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    predictor = build_sam2_video_predictor(str(args.config), str(args.checkpoint), device=device)

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        state = predictor.init_state(video_path=str(args.frames_dir))
        for prompt_frame_idx in sorted(prompts_by_frame):
            prompt_values = prompts_by_frame[prompt_frame_idx]
            points = np.array([[x, y] for x, y, _label in prompt_values], dtype=np.float32)
            labels = np.array([label for _x, _y, label in prompt_values], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt_frame_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )

        saved_rows = []
        for sam_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=propagate_start_idx,
            max_frame_num_to_track=args.max_track_frames,
            reverse=args.reverse,
        ):
            mask = (mask_logits[0, 0] > float(args.threshold)).detach().cpu().numpy().astype(np.uint8) * 255
            bag_frame_id = int(frame_map[int(sam_idx)]["bag_frame_id"])
            mask_path = mask_dir / f"mask_{bag_frame_id:06d}.png"
            cv2.imwrite(str(mask_path), mask)

            image = cv2.imread(str(args.frames_dir / f"{int(sam_idx):05d}.jpg"), cv2.IMREAD_COLOR)
            if image is not None:
                blue = np.zeros_like(image)
                blue[:, :, 0] = 255
                overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(image, 0.45, blue, 0.55, 0), image)
                cv2.circle(overlay, (int(args.click_x), int(args.click_y)), 5, (0, 255, 255), -1)
                cv2.putText(
                    overlay,
                    f"bag frame {bag_frame_id}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(overlay_dir / f"mask_overlay_{bag_frame_id:06d}.png"), overlay)

            saved_rows.append(
                {
                    "sam_frame_idx": int(sam_idx),
                    "bag_frame_id": bag_frame_id,
                    "mask": str(mask_path.as_posix()),
                    "mask_pixels": int(np.count_nonzero(mask)),
                }
            )

    (args.out_dir / "mask_map.json").write_text(json.dumps(saved_rows, indent=2), encoding="utf-8")
    meta = {
        "frames_dir": str(args.frames_dir.as_posix()),
        "frame_map": str(args.frame_map.as_posix()),
        "checkpoint": str(args.checkpoint.as_posix()),
        "config": args.config,
        "device": device,
        "click_sam_idx": click_sam_idx,
        "click_frame_input": args.click_frame,
        "click_x": args.click_x,
        "click_y": args.click_y,
        "prompts_by_sam_frame": {
            str(frame_idx): [
                {"x": x, "y": y, "label": label}
                for x, y, label in values
            ]
            for frame_idx, values in prompts_by_frame.items()
        },
        "threshold": args.threshold,
        "reverse": args.reverse,
        "propagate_start_idx": propagate_start_idx,
        "max_track_frames": args.max_track_frames,
    }
    (args.out_dir / "sam2_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {len(saved_rows)} masks to {mask_dir}")
    print(f"saved overlays to {overlay_dir}")


if __name__ == "__main__":
    # Avoid OpenMP duplicate-library crashes sometimes seen on Windows/PyTorch.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
