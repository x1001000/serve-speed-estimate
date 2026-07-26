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


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    # Pad with edge values (not zeros) so the endpoints aren't dragged toward 0,
    # which would create large artificial jumps at the start/end of the path.
    half_lo = (window - 1) // 2
    half_hi = window // 2
    padded = np.pad(values, (half_lo, half_hi), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


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
    smooth_window: int = 3,
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

    xs = _moving_average(x, smooth_window)
    ys = _moving_average(y, smooth_window)

    dt = np.diff(t)
    valid = dt > 1e-6
    dx = np.diff(xs)[valid]
    dy = np.diff(ys)[valid]
    dt = dt[valid]
    seg_mid_t = ((t[:-1] + t[1:]) / 2.0)[valid]

    if len(dt) == 0:
        raise ValueError("Trajectory has no usable time gaps between detections.")

    seg_dist_m = np.hypot(dx, dy) * mpp
    seg_speed = seg_dist_m / dt
    seg_speed_f = _median_filter(seg_speed, window=3)

    peak = float(np.max(seg_speed_f))
    # Average over the "flight" portion: segments at least half the peak speed.
    flight_mask = seg_speed_f >= 0.5 * peak
    avg = float(np.mean(seg_speed_f[flight_mask])) if flight_mask.any() else float(np.mean(seg_speed_f))

    path_length_m = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)) * mpp))
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
