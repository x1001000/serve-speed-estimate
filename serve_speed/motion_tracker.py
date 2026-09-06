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
) -> list[tuple[int, float, float]]:
    """Follow the ball in one time direction (``step`` = +1 forward, -1 back)."""
    max_disp = max_speed_ms / fps / meters_per_pixel  # max plausible px / frame
    path = [(seed_frame, float(seed_xy[0]), float(seed_xy[1]))]
    vx = vy = 0.0
    dir_sign = 0  # established horizontal travel direction
    miss = 0
    i = seed_frame + step
    while 0 <= i < n_frames:
        gap = abs(i - path[-1][0])
        px, py = path[-1][1], path[-1][2]
        predx, predy = px + vx * step, py + vy * step
        cands = blobs_by_frame.get(i, [])
        radius = max(search_min_px, search_k * float(np.hypot(vx, vy)))
        near = [
            (cx, cy)
            for cx, cy in cands
            if np.hypot(cx - predx, cy - predy) <= radius
            and np.hypot(cx - px, cy - py) <= max_disp * gap
        ]
        if not near:
            miss += 1
            if miss > max_miss:
                break
            i += step
            continue
        miss = 0
        cx, cy = min(near, key=lambda p: np.hypot(p[0] - predx, p[1] - predy))
        dx = (cx - px) / gap
        # Stop if the ball clearly reverses horizontal direction (received/bounced).
        new_sign = np.sign(dx)
        if dir_sign != 0 and new_sign != 0 and new_sign != dir_sign:
            break
        if abs(dx) > 1.0:
            dir_sign = new_sign
        vx, vy = dx, (cy - py) / gap
        path.append((i, cx, cy))
        i += step
    return path


def _ballistic_trim(path: list[tuple[int, float, float]], max_residual_px: float) -> list[tuple[int, float, float]]:
    """Keep the longest contiguous run that fits a projectile (x linear, y quad)."""
    if len(path) < 4:
        return path
    path = sorted(path, key=lambda p: p[0])
    f = np.array([p[0] for p in path], float)
    x = np.array([p[1] for p in path], float)
    y = np.array([p[2] for p in path], float)
    f0 = f - f.mean()
    xr = np.abs(x - np.polyval(np.polyfit(f0, x, 1), f0))
    yr = np.abs(y - np.polyval(np.polyfit(f0, y, 2), f0))
    ok = (np.hypot(xr, yr) <= max_residual_px).tolist()
    # longest run of True
    best_s = best_e = cur_s = None
    best = 0
    for idx, good in enumerate(ok + [False]):
        if good and cur_s is None:
            cur_s = idx
        elif not good and cur_s is not None:
            if idx - cur_s > best:
                best, best_s, best_e = idx - cur_s, cur_s, idx
            cur_s = None
    if best_s is None:
        return path
    return path[best_s:best_e]


def track_serve(
    gray: list[np.ndarray],
    seed_frame: int,
    seed_xy: tuple[float, float],
    fps: float,
    meters_per_pixel: float,
    max_speed_ms: float = 45.0,
    thresh: int = 18,
    search_min_px: float = 30.0,
    search_k: float = 2.5,
    max_miss: int = 2,
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
        search_k=search_k, max_miss=max_miss,
    )
    fwd = _follow(blobs_by_frame, seed_frame, seed, +1, **kw)
    bwd = _follow(blobs_by_frame, seed_frame, seed, -1, **kw)

    merged = {p[0]: p for p in bwd}
    merged.update({p[0]: p for p in fwd})  # seed frame shared; either is fine
    path = _ballistic_trim(sorted(merged.values(), key=lambda p: p[0]), max_residual_px)

    return [TrackPoint(f, f / fps, x, y, tracker_id=1, conf=1.0) for f, x, y in path]
