"""End-to-end serve-speed pipeline: video -> trajectory -> speed estimate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .detector import get_detector
from .speed import Calibration, SpeedEstimate, estimate_speed
from .tracking import (
    TrackPoint,
    build_tracker,
    detections_to_points,
    select_serve_trajectory,
)
from .video import iter_frames, video_info

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineResult:
    estimate: SpeedEstimate
    trajectory: list[TrackPoint]
    ball_boxes: dict[int, tuple[float, float, float, float]]
    fps: float


def run_pipeline(
    video_path: str,
    calibration: Calibration,
    model_name: str = "large",
    threshold: float = 0.4,
    stride: int = 1,
    tracker_name: str = "ocsort",
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Detect and track the ball across the clip and estimate the serve speed."""
    info = video_info(video_path)
    fps = info["fps"]
    total = max(1, info["frame_count"])

    detector = get_detector(model_name=model_name)
    tracker = build_tracker(tracker_name)

    all_points: list[TrackPoint] = []
    # Best single ball box per source frame index, for annotation.
    ball_boxes: dict[int, tuple[float, float, float, float]] = {}

    for frame_idx, frame in iter_frames(video_path, stride=stride):
        if progress is not None:
            progress(min(0.95, frame_idx / total), "Detecting & tracking ball…")

        boxes, detections = detector.detect(frame, threshold=threshold)

        # Feed *all* ball detections to the tracker so it can form proper
        # tracks; the serve is then isolated by physical plausibility, not by
        # blindly trusting the single most confident box (which jumps between
        # players, a ball on the floor, wall pads, etc.).
        tracked = tracker.update(detections)
        t = frame_idx / fps
        all_points.extend(detections_to_points(tracked, frame_idx, t))

    if progress is not None:
        progress(0.96, "Estimating speed…")

    trajectory = select_serve_trajectory(
        all_points, fps=fps, meters_per_pixel=calibration.meters_per_pixel
    )
    if len(trajectory) < 2:
        raise ValueError(
            "Couldn't isolate a clean serve trajectory. The ball is likely too "
            "small/blurred to detect reliably at this camera distance, or the "
            "clip has too much other motion. Try a closer or zoomed-in view "
            "(ball at least ~20 px), a higher frame rate, or trim the clip to "
            "just the serve."
        )

    # Reconstruct small ball boxes at the chosen trajectory points for overlay
    # (a volleyball is ~0.21 m across).
    radius_px = max(3.0, 0.105 / calibration.meters_per_pixel)
    for p in trajectory:
        ball_boxes[p.frame_idx] = (
            p.x - radius_px, p.y - radius_px, p.x + radius_px, p.y + radius_px,
        )

    estimate = estimate_speed(trajectory, calibration)

    return PipelineResult(
        estimate=estimate,
        trajectory=trajectory,
        ball_boxes=ball_boxes,
        fps=fps,
    )
