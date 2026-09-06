"""End-to-end serve-speed pipeline: click + video -> trajectory -> speed.

The ball is followed by a click-seeded motion-ballistic tracker (see
``motion_tracker``) rather than an appearance detector, because a served ball
is often only a few pixels across — too small for a generic detector, but easy
for frame-difference motion tracking once the user's click says which blob to
follow. This runs on CPU; no GPU or model download is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .motion_tracker import track_serve
from .speed import Calibration, SpeedEstimate, estimate_speed
from .tracking import TrackPoint
from .video import read_gray_frames

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
    seed_frame: int,
    seed_xy: tuple[float, float],
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Follow the ball from the clicked seed and estimate the serve speed."""
    if progress is not None:
        progress(0.1, "Reading frames…")
    gray, fps = read_gray_frames(video_path)
    if len(gray) < 3:
        raise ValueError("The clip is too short to analyse.")

    if progress is not None:
        progress(0.5, "Tracking the ball…")
    seed_frame = max(0, min(int(seed_frame), len(gray) - 1))
    trajectory = track_serve(
        gray,
        seed_frame=seed_frame,
        seed_xy=(float(seed_xy[0]), float(seed_xy[1])),
        fps=fps,
        meters_per_pixel=calibration.meters_per_pixel,
    )
    if len(trajectory) < 3:
        raise ValueError(
            "Couldn't follow the ball from the clicked point. Click directly on "
            "the ball while it is in flight (on a frame where you can see it), "
            "and make sure the clip actually shows the ball moving."
        )

    # Small ball boxes at each tracked point for the overlay (~0.21 m ball).
    radius_px = max(4.0, 0.105 / calibration.meters_per_pixel)
    ball_boxes = {
        p.frame_idx: (p.x - radius_px, p.y - radius_px, p.x + radius_px, p.y + radius_px)
        for p in trajectory
    }

    if progress is not None:
        progress(0.85, "Estimating speed…")
    estimate = estimate_speed(trajectory, calibration)

    return PipelineResult(
        estimate=estimate,
        trajectory=trajectory,
        ball_boxes=ball_boxes,
        fps=fps,
    )
