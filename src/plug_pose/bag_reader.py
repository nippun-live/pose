from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class StreamSummary:
    stream: str
    format: str
    width: int | None
    height: int | None
    fps: int | None
    index: int


@dataclass(frozen=True)
class FrameData:
    frame_id: int
    timestamp_s: float
    color_rgb: np.ndarray
    depth_z16: np.ndarray


def start_playback(bag_path: str | Path) -> tuple[rs.pipeline, rs.pipeline_profile]:
    bag_path = Path(bag_path).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    return pipeline, profile


def stream_summaries(profile: rs.pipeline_profile) -> list[StreamSummary]:
    summaries: list[StreamSummary] = []
    for stream_profile in profile.get_streams():
        video_profile = stream_profile.as_video_stream_profile()
        summaries.append(
            StreamSummary(
                stream=str(stream_profile.stream_type()).replace("stream.", ""),
                format=str(stream_profile.format()).replace("format.", ""),
                width=video_profile.width(),
                height=video_profile.height(),
                fps=stream_profile.fps(),
                index=stream_profile.stream_index(),
            )
        )
    return summaries


def get_color_intrinsics(profile: rs.pipeline_profile) -> dict:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def get_depth_scale(profile: rs.pipeline_profile) -> float | None:
    device = profile.get_device()
    for sensor in device.query_sensors():
        if sensor.is_depth_sensor():
            return sensor.as_depth_sensor().get_depth_scale()
    return None


def intrinsics_to_camera_matrix(intrinsics: dict) -> np.ndarray:
    return np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def intrinsics_to_dist_coeffs(intrinsics: dict) -> np.ndarray:
    return np.array(intrinsics.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)


def save_intrinsics(path: str | Path, intrinsics: dict, depth_scale: float | None) -> None:
    payload = {
        "color_intrinsics": intrinsics,
        "depth_scale_m_per_unit": depth_scale,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_intrinsics(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["color_intrinsics"]


def iter_aligned_frames(
    bag_path: str | Path,
    max_frames: int | None = None,
    timeout_ms: int = 5000,
) -> Iterator[FrameData]:
    pipeline, _profile = start_playback(bag_path)
    align = rs.align(rs.stream.color)
    frame_id = 0
    first_timestamp_ms: float | None = None

    try:
        while max_frames is None or frame_id < max_frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
            except RuntimeError:
                break

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            timestamp_ms = float(frames.get_timestamp())
            if first_timestamp_ms is None:
                first_timestamp_ms = timestamp_ms

            yield FrameData(
                frame_id=frame_id,
                timestamp_s=(timestamp_ms - first_timestamp_ms) / 1000.0,
                color_rgb=color.copy(),
                depth_z16=depth.copy(),
            )
            frame_id += 1
    finally:
        pipeline.stop()


def summaries_as_dicts(summaries: list[StreamSummary]) -> list[dict]:
    return [asdict(summary) for summary in summaries]
