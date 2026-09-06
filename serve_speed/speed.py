"""Turn a pixel-space ball trajectory into a real-world speed estimate.

Calibration is a single scalar: the user marks a segment of known real length
in the image (by default the 18 m court end-line to end-line distance) which
gives a metres-per-pixel scale.  This assumes the ball travels roughly in the
plane of that reference line.  For a camera at the side of the court that is a
reasonable approximation and is what makes the estimate a genuine *estimate*
rather than a precise measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tracking import TrackPoint


@dataclass
class Calibration:
    """Pixel-to-metre mapping from a marked reference segment."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    real_distance_m: float = 18.0

    @property
    def pixel_distance(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))

    @property
    def meters_per_pixel(self) -> float:
        px = self.pixel_distance
        if px <= 1e-6:
            raise ValueError("Calibration points are identical; mark two distinct points.")
        return self.real_distance_m / px


@dataclass
class SpeedEstimate:
    peak_ms: float
    avg_ms: float
    meters_per_pixel: float
    n_points: int
    duration_s: float
    path_length_m: float
    times: np.ndarray = field(repr=False)
    speeds_ms: np.ndarray = field(repr=False)
    trajectory_px: np.ndarray = field(repr=False)

    @property
    def peak_kmh(self) -> float:
        return self.peak_ms * 3.6

    @property
    def avg_kmh(self) -> float:
        return self.avg_ms * 3.6


def _median_filter(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array(
        [np.median(padded[i : i + window]) for i in range(len(values))]
    )


def estimate_speed(
    points: list[TrackPoint],
    calibration: Calibration,
    max_speed_ms: float = 45.0,
) -> SpeedEstimate:
    """Estimate serve speed from an ordered ball trajectory.

    Returns both a peak speed (the fastest part of the flight, closest to the
    speed just after contact) and an average speed over the fast portion of
    the trajectory.
    """
    if len(points) < 2:
        raise ValueError(
            "Not enough ball detections to estimate speed. Try a clearer clip, "
            "a smaller detection threshold, or a larger model."
        )

    mpp = calibration.meters_per_pixel

    t = np.array([p.t for p in points], dtype=float)
    x = np.array([p.x for p in points], dtype=float)
    y = np.array([p.y for p in points], dtype=float)
    traj_px = np.column_stack([x, y])

    # Speeds from *raw* positions. Do NOT smooth positions here: the samples are
    # non-uniform in time (the ball's blob drops out for several frames at the
    # apex / low-contrast background), and index-based smoothing would blend
    # points across those gaps and manufacture huge fake speeds. Each segment
    # speed is dist/dt, which already handles uneven gaps correctly.
    dt = np.diff(t)
    valid = dt > 1e-6
    dx = np.diff(x)[valid]
    dy = np.diff(y)[valid]
    dt = dt[valid]
    seg_mid_t = ((t[:-1] + t[1:]) / 2.0)[valid]

    if len(dt) == 0:
        raise ValueError("Trajectory has no usable time gaps between detections.")

    seg_speed = np.hypot(dx, dy) * mpp / dt
    # De-jitter the *speed* series (centroid noise of a few px/frame), not the
    # positions.
    seg_speed_f = _median_filter(seg_speed, window=3)

    # Defensive cap: no volleyball serve exceeds ~45 m/s (162 km/h; the men's
    # record is ~37 m/s). Anything above that is a detection jump, not the ball.
    plausible = seg_speed_f[seg_speed_f <= max_speed_ms]
    if plausible.size == 0:
        raise ValueError(
            "Every tracked segment implies an impossible speed — the track is "
            "jumping between objects rather than following the ball. Click "
            "directly on the ball while it is clearly in flight."
        )

    # Robust peak: 90th percentile of plausible segments (ignores a lone noisy
    # frame) — closest to the true speed just after contact.
    peak = float(np.percentile(plausible, 90))
    flight_mask = (seg_speed_f >= 0.5 * peak) & (seg_speed_f <= max_speed_ms)
    avg = float(np.mean(seg_speed_f[flight_mask])) if flight_mask.any() else float(np.mean(plausible))

    path_length_m = float(np.sum(np.hypot(dx, dy) * mpp))
    duration = float(t[-1] - t[0])

    return SpeedEstimate(
        peak_ms=peak,
        avg_ms=avg,
        meters_per_pixel=mpp,
        n_points=len(points),
        duration_s=duration,
        path_length_m=path_length_m,
        times=seg_mid_t,
        speeds_ms=seg_speed_f,
        trajectory_px=traj_px,
    )
