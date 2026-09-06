"""Click-seeded motion-ballistic ball tracker.

A served volleyball is tiny (often only a handful of pixels in a wide court
shot) and motion-blurred — appearance detectors like RF-DETR miss it and fire
on everything round instead. But a small fast ball is a *bright moving blob*,
which frame-differencing spots reliably. The only hard part is knowing *which*
blob is the ball among all the moving people; the user's single click answers
that, and a projectile motion model carries it forward frame to frame.

The tracker therefore:
1. finds moving blobs per frame by three-frame differencing,
2. starts from the clicked ball position,
3. follows the blob nearest the *predicted* next position (so large per-frame
   jumps are fine — we predict where to look),
4. stops when the flight ends (direction reversal / lost / impossible jump),
5. trims the result to its clean ballistic span.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .tracking import TrackPoint

logger = logging.getLogger(__name__)


def motion_blobs(
    prev: np.ndarray,
    cur: np.ndarray,
    nxt: np.ndarray,
    thresh: int = 18,
    min_area: float = 3.0,
    max_area: float = 250.0,
    min_aspect: float = 0.3,
    max_aspect: float = 3.2,
) -> list[tuple[float, float]]:
    """Centroids of small compact blobs moving in BOTH adjacent frame gaps.

    AND-ing ``|cur-prev|`` with ``|nxt-cur|`` keeps only things moving *through*
    this frame, which suppresses one-off flicker and static edges.
    """
    d1 = cv2.absdiff(cur, prev)
    d2 = cv2.absdiff(nxt, cur)
    mv = cv2.threshold(cv2.bitwise_and(d1, d2), thresh, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(mv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[float, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if min_area <= area <= max_area and min_aspect <= aspect <= max_aspect:
            out.append((x + w / 2.0, y + h / 2.0))
    return out


def _all_blobs(gray: list[np.ndarray], thresh: int) -> dict[int, list[tuple[float, float]]]:
    return {
        i: motion_blobs(gray[i - 1], gray[i], gray[i + 1], thresh=thresh)
        for i in range(1, len(gray) - 1)
    }


def _predict(path: list[tuple[int, float, float]], frame: int, fit_window: int) -> tuple[float, float]:
    """Predict the ball position at ``frame`` from the accepted points.

    With enough points we extrapolate a projectile (x linear, y quadratic in
    frame index), which curves correctly across gaps; otherwise fall back to
    constant velocity, then to the last point.
    """
    pts = path[-fit_window:]
    f = np.array([p[0] for p in pts], float)
    x = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    f0 = f.mean()
    # Only trust a quadratic (gravity) fit once we have enough spread; near the
    # apex a quadratic over few flat points extrapolates wildly, whereas
    # constant velocity is stable. Below the threshold, use linear prediction.
    if len(pts) >= 6:
        px = np.polyval(np.polyfit(f - f0, x, 1), frame - f0)
        py = np.polyval(np.polyfit(f - f0, y, 2), frame - f0)
        return float(px), float(py)
    if len(pts) >= 2:
        dt = f[-1] - f[-2]
        vx = (x[-1] - x[-2]) / dt
        vy = (y[-1] - y[-2]) / dt
        return float(x[-1] + vx * (frame - f[-1])), float(y[-1] + vy * (frame - f[-1]))
    return float(x[-1]), float(y[-1])


def _follow(
    blobs_by_frame: dict[int, list[tuple[float, float]]],
    seed_frame: int,
    seed_xy: tuple[float, float],
    step: int,
    n_frames: int,
    fps: float,
    meters_per_pixel: float,
    max_speed_ms: float,
    search_min_px: float,
    search_k: float,
    max_miss: int,
    fit_window: int,
) -> list[tuple[int, float, float]]:
    """Follow the ball in one time direction (``step`` = +1 forward, -1 back).

    The ball's motion blob routinely vanishes for several frames (apex, low
    contrast, occlusion by players). We therefore *coast* across gaps using the
    projectile prediction and only give up after ``max_miss`` consecutive empty
    frames — the parabola keeps the search window on the ball's real path.
    """
    max_disp = max_speed_ms / fps / meters_per_pixel  # max plausible px / frame
    path = [(seed_frame, float(seed_xy[0]), float(seed_xy[1]))]
    miss = 0
    i = seed_frame + step
    while 0 <= i < n_frames:
        lastf, lx, ly = path[-1]
        gap = abs(i - lastf)
        predx, predy = _predict(path, i, fit_window)
        # Search window grows a little while coasting (prediction less certain).
        radius = search_min_px + search_k * miss
        cands = blobs_by_frame.get(i, [])
        near = [
            (cx, cy)
            for cx, cy in cands
            if np.hypot(cx - predx, cy - predy) <= radius
            and np.hypot(cx - lx, cy - ly) <= max_disp * gap
        ]
        if not near:
            miss += 1
            if miss > max_miss:
                break
            i += step
            continue
        miss = 0
        cx, cy = min(near, key=lambda p: np.hypot(p[0] - predx, p[1] - predy))
        path.append((i, cx, cy))
        i += step
    return path


def _projectile_residuals(f, x, y, sample):
    """Residuals of all points to a projectile fit through ``sample`` indices."""
    f0 = f.mean()
    fs = f[sample]
    cx = np.polyfit(fs - f0, x[sample], 1)  # x linear
    cy = np.polyfit(fs - f0, y[sample], 2)  # y quadratic (gravity)
    return np.hypot(x - np.polyval(cx, f - f0), y - np.polyval(cy, f - f0))


def _ballistic_trim(
    path: list[tuple[int, float, float]],
    max_residual_px: float,
    seed_frame: int,
) -> list[tuple[int, float, float]]:
    """Keep the largest projectile-consistent set of points (RANSAC).

    The raw track can pick up clutter before the serve and after the ball is
    received (landing-area blobs), so a single global fit is not robust. RANSAC
    finds the projectile with the most inliers; we then refit on those inliers
    and keep the contiguous-in-time run that contains the clicked seed frame.
    """
    if len(path) < 5:
        return path
    path = sorted(path, key=lambda p: p[0])
    f = np.array([p[0] for p in path], float)
    x = np.array([p[1] for p in path], float)
    y = np.array([p[2] for p in path], float)
    n = len(path)

    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    for _ in range(300):
        s = rng.choice(n, 3, replace=False)
        if len(set(f[s].tolist())) < 3:
            continue
        inliers = _projectile_residuals(f, x, y, s) <= max_residual_px
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers
    if best_inliers is None or best_count < 4:
        return path

    # Refit on inliers for a stable model, then re-threshold.
    keep = _projectile_residuals(f, x, y, np.flatnonzero(best_inliers)) <= max_residual_px

    # Contiguous run (in the time-sorted list) containing the seed frame.
    seed_pos = int(np.argmin(np.abs(f - seed_frame)))
    if not keep[seed_pos]:
        return [path[i] for i in range(n) if keep[i]]
    lo = hi = seed_pos
    while lo - 1 >= 0 and keep[lo - 1]:
        lo -= 1
    while hi + 1 < n and keep[hi + 1]:
        hi += 1
    return path[lo : hi + 1]


def track_serve(
    gray: list[np.ndarray],
    seed_frame: int,
    seed_xy: tuple[float, float],
    fps: float,
    meters_per_pixel: float,
    max_speed_ms: float = 45.0,
    thresh: int = 18,
    search_min_px: float = 40.0,
    search_k: float = 1.0,
    max_miss: int = 12,
    fit_window: int = 8,
    snap_radius_px: float = 40.0,
    max_residual_px: float = 12.0,
) -> list[TrackPoint]:
    """Track the ball from a clicked seed and return its clean flight path."""
    n = len(gray)
    if n < 3 or not (0 <= seed_frame < n):
        return []
    blobs_by_frame = _all_blobs(gray, thresh)

    # Snap the click to the nearest moving blob (the user won't hit the 5px ball
    # exactly). Fall back to the raw click if nothing is moving nearby.
    seed = (float(seed_xy[0]), float(seed_xy[1]))
    near_seed = blobs_by_frame.get(seed_frame, [])
    if near_seed:
        c = min(near_seed, key=lambda p: np.hypot(p[0] - seed[0], p[1] - seed[1]))
        if np.hypot(c[0] - seed[0], c[1] - seed[1]) <= snap_radius_px:
            seed = c

    kw = dict(
        n_frames=n, fps=fps, meters_per_pixel=meters_per_pixel,
        max_speed_ms=max_speed_ms, search_min_px=search_min_px,
        search_k=search_k, max_miss=max_miss, fit_window=fit_window,
    )
    fwd = _follow(blobs_by_frame, seed_frame, seed, +1, **kw)
    bwd = _follow(blobs_by_frame, seed_frame, seed, -1, **kw)

    merged = {p[0]: p for p in bwd}
    merged.update({p[0]: p for p in fwd})  # seed frame shared; either is fine
    path = _ballistic_trim(
        sorted(merged.values(), key=lambda p: p[0]), max_residual_px, seed_frame
    )

    return [TrackPoint(f, f / fps, x, y, tracker_id=1, conf=1.0) for f, x, y in path]
