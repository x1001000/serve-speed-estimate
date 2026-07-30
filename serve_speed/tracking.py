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


def build_tracker(name: str = "bytetrack"):
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


def select_serve_trajectory(
    points: list[TrackPoint], min_points: int = 4
) -> list[TrackPoint]:
    """Pick the single track that best represents the serve.

    The serve is the fastest, longest-travelling ball motion in the clip, so
    we score each track by the total pixel distance its centre travels and
    return the winner's points ordered in time.
    """
    if not points:
        return []

    by_track: dict[int, list[TrackPoint]] = {}
    for p in points:
        by_track.setdefault(p.tracker_id, []).append(p)

    def travel(track: list[TrackPoint]) -> float:
        track = sorted(track, key=lambda p: p.frame_idx)
        d = 0.0
        for a, b in zip(track, track[1:]):
            d += float(np.hypot(b.x - a.x, b.y - a.y))
        return d

    candidates = [t for t in by_track.values() if len(t) >= min_points]
    if not candidates:
        # Fall back to the longest available track even if short.
        candidates = [max(by_track.values(), key=len)]

    best = max(candidates, key=travel)
    return sorted(best, key=lambda p: p.frame_idx)
