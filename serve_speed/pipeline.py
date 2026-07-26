"""End-to-end serve-speed pipeline: video -> trajectory -> speed estimate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .detector import BallDetector
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
    model_name: str = "medium",
    threshold: float = 0.4,
    stride: int = 1,
    tracker_name: str = "sort",
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Detect and track the ball across the clip and estimate the serve speed."""
    info = video_info(video_path)
    fps = info["fps"]
    total = max(1, info["frame_count"])

    detector = BallDetector(model_name=model_name)
    tracker = build_tracker(tracker_name)

    all_points: list[TrackPoint] = []
    # Best single ball box per source frame index, for annotation.
    ball_boxes: dict[int, tuple[float, float, float, float]] = {}

    for frame_idx, frame in iter_frames(video_path, stride=stride):
        if progress is not None:
            progress(min(0.95, frame_idx / total), "Detecting & tracking ball…")

        boxes, detections = detector.detect(frame, threshold=threshold)

        # Feed *only* the single best ball detection to the tracker: on a
        # side-view court there is one ball, and this keeps the trajectory
        # clean and cheap.
        if len(detections) > 1:
            detections = detections[:1]

        tracked = tracker.update(detections)
        t = frame_idx / fps
        pts = detections_to_points(tracked, frame_idx, t)
        all_points.extend(pts)

        if len(boxes) > 0:
            x1, y1, x2, y2, _ = boxes[0]
            ball_boxes[frame_idx] = (float(x1), float(y1), float(x2), float(y2))

    if progress is not None:
        progress(0.96, "Estimating speed…")

    trajectory = select_serve_trajectory(all_points)
    estimate = estimate_speed(trajectory, calibration)

    # Keep only ball boxes that belong to the selected trajectory frames.
    traj_frames = {p.frame_idx for p in trajectory}
    ball_boxes = {k: v for k, v in ball_boxes.items() if k in traj_frames}

    return PipelineResult(
        estimate=estimate,
        trajectory=trajectory,
        ball_boxes=ball_boxes,
        fps=fps,
    )
