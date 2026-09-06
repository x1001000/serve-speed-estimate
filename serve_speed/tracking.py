"""Multi-frame ball tracking built on the roboflow ``trackers`` package.

We link per-frame ball detections into tracks so that spurious one-off
detections can be discarded and a single, temporally-consistent trajectory
(the serve) can be isolated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TRACKER_CLASSES = {
    "sort": "SORTTracker",
    "bytetrack": "ByteTrackTracker",
    "ocsort": "OCSORTTracker",
}


@dataclass
class TrackPoint:
    """One observation of the ball along the trajectory."""

    frame_idx: int
    t: float  # seconds
    x: float  # pixel x of ball centre
    y: float  # pixel y of ball centre
    tracker_id: int
    conf: float


def build_tracker(name: str = "ocsort"):
    """Instantiate a tracker from the ``trackers`` package.

    Falls back gracefully to whatever tracker class is importable so the app
    keeps working across ``trackers`` releases.
    """
    import trackers as _trackers

    for candidate in (_TRACKER_CLASSES.get(name, "SORTTracker"), "SORTTracker", "ByteTrackTracker"):
        cls = getattr(_trackers, candidate, None)
        if cls is not None:
            logger.info("Using tracker %s", candidate)
            return cls()
    raise ImportError("No usable tracker class found in the 'trackers' package.")


def detections_to_points(detections, frame_idx: int, t: float) -> list[TrackPoint]:
    """Convert a tracked ``supervision.Detections`` object to ``TrackPoint``s."""
    xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
    if len(xyxy) == 0:
        return []
    tracker_id = getattr(detections, "tracker_id", None)
    conf = getattr(detections, "confidence", None)

    points: list[TrackPoint] = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        tid = int(tracker_id[i]) if tracker_id is not None else -1
        c = float(conf[i]) if conf is not None and len(conf) > i else 1.0
        points.append(
            TrackPoint(
                frame_idx=frame_idx,
                t=t,
                x=(x1 + x2) / 2.0,
                y=(y1 + y2) / 2.0,
                tracker_id=tid,
                conf=c,
            )
        )
    return points


def _fit_ballistic(track: list[TrackPoint]) -> float:
    """RMS residual (px) of a projectile fit: x linear, y quadratic vs frame.

    A real serve is a projectile — horizontal position is ~linear in time and
    vertical position is ~quadratic (gravity). Clutter (players, a ball on the
    floor, wall pads jumped between by the detector) does not fit this model,
    so a low residual is a strong "this is really the ball" signal.
    """
    f = np.array([p.frame_idx for p in track], dtype=float)
    x = np.array([p.x for p in track], dtype=float)
    y = np.array([p.y for p in track], dtype=float)
    f0 = f - f.mean()  # centre for numerical stability
    xr = x - np.polyval(np.polyfit(f0, x, 1), f0)
    yr = y - np.polyval(np.polyfit(f0, y, 2), f0)
    return float(np.sqrt(np.mean(xr**2 + yr**2)))


def _split_by_motion(
    track: list[TrackPoint],
    fps: float,
    meters_per_pixel: float,
    max_speed_ms: float,
    max_gap_s: float,
) -> list[list[TrackPoint]]:
    """Split a time-sorted track wherever a link is physically impossible.

    A ball cannot teleport: any consecutive pair implying a speed above
    ``max_speed_ms`` (or separated by more than ``max_gap_s``) is a detection
    jump/ID-switch, so we cut there. A zig-zag "track" collapses into a bunch
    of tiny fragments that are then rejected for being too short.
    """
    track = sorted(track, key=lambda p: p.frame_idx)
    segments: list[list[TrackPoint]] = []
    current = [track[0]]
    for a, b in zip(track, track[1:]):
        dt = (b.frame_idx - a.frame_idx) / fps
        if dt <= 0:
            continue
        speed = float(np.hypot(b.x - a.x, b.y - a.y)) * meters_per_pixel / dt
        if dt > max_gap_s or speed > max_speed_ms:
            segments.append(current)
            current = [b]
        else:
            current.append(b)
    segments.append(current)
    return segments


def select_serve_trajectory(
    points: list[TrackPoint],
    fps: float,
    meters_per_pixel: float,
    max_speed_ms: float = 45.0,
    min_points: int = 5,
    max_gap_s: float = 0.34,
    max_residual_px: float = 12.0,
    min_horizontal_m: float = 1.0,
) -> list[TrackPoint]:
    """Extract the single ball trajectory that actually looks like a serve.

    Unlike a naive "longest travel" pick (which rewards zig-zagging noise),
    this keeps only trajectory pieces that are *physically plausible*: bounded
    frame-to-frame speed, a good projectile fit, and real horizontal travel.
    Returns ``[]`` when nothing qualifies, so the caller can report low
    confidence rather than emit a nonsense number.
    """
    if not points or fps <= 0 or meters_per_pixel <= 0:
        return []

    by_track: dict[int, list[TrackPoint]] = {}
    for p in points:
        by_track.setdefault(p.tracker_id, []).append(p)

    scored: list[tuple[float, list[TrackPoint]]] = []
    for track in by_track.values():
        for seg in _split_by_motion(track, fps, meters_per_pixel, max_speed_ms, max_gap_s):
            if len(seg) < min_points:
                continue
            horizontal_m = abs(seg[-1].x - seg[0].x) * meters_per_pixel
            if horizontal_m < min_horizontal_m:
                continue  # a serve crosses the court; near-stationary blobs don't
            residual = _fit_ballistic(seg)
            if residual > max_residual_px:
                continue  # doesn't follow a projectile path -> not the ball
            frames = [p.frame_idx for p in seg]
            span = max(frames) - min(frames)
            # Prefer more points and longer span, penalise a poor fit.
            score = len(seg) + 0.05 * span - 0.1 * residual
            scored.append((score, seg))

    if not scored:
        return []
    scored.sort(key=lambda s: s[0], reverse=True)
    return sorted(scored[0][1], key=lambda p: p.frame_idx)
