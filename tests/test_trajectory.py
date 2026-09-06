"""Trajectory-extraction tests.

These guard the property that broke on real footage: the old selector picked
the track with the largest *total pixel travel*, which is exactly the zig-zag
of false-positive detections a generic detector emits for a tiny, fast ball in
a crowded gym — producing absurd speeds (e.g. 277 km/h). The extractor must
instead pick the physically plausible (projectile) path, and report nothing
when no such path exists.
"""

import numpy as np

from serve_speed.speed import Calibration, estimate_speed
from serve_speed.tracking import TrackPoint, select_serve_trajectory

FPS = 30.0
# Calibration matching the example clip: 18 m across ~620 px.
CALIB = Calibration(p1=(200, 395), p2=(820, 378), real_distance_m=18.0)
MPP = CALIB.meters_per_pixel


def _make_points(seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    # Real serve (id=1): ~18 frames, parabola, ~70 km/h (19.4 m/s).
    vx_px = 19.4 / FPS / MPP
    for k in range(18):
        f = 20 + k
        x = 250 + vx_px * k
        y = 300 - 6 * k + 0.5 * k * k
        pts.append(TrackPoint(f, f / FPS, x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 1, 0.7))
    # Zig-zag distractor (id=2): the old failure mode.
    for k in range(30):
        pts.append(TrackPoint(10 + k, (10 + k) / FPS, rng.uniform(300, 850), rng.uniform(250, 400), 2, 0.6))
    # Ball sitting on the floor (id=3): barely moves.
    for k in range(25):
        pts.append(TrackPoint(5 + k, (5 + k) / FPS, 360 + rng.normal(0, 2), 395 + rng.normal(0, 2), 3, 0.8))
    # One-off false positives.
    for k in range(40):
        pts.append(TrackPoint(int(rng.integers(0, 116)), 0.0, rng.uniform(0, 960), rng.uniform(0, 540), 100 + k, 0.5))
    return pts


def test_extracts_serve_from_clutter():
    traj = select_serve_trajectory(_make_points(), fps=FPS, meters_per_pixel=MPP)
    assert {p.tracker_id for p in traj} == {1}
    est = estimate_speed(traj, CALIB)
    assert 55 < est.peak_kmh < 90  # synthetic serve is ~70 km/h


def test_no_serve_in_pure_clutter():
    clutter = [p for p in _make_points() if p.tracker_id != 1]
    assert select_serve_trajectory(clutter, fps=FPS, meters_per_pixel=MPP) == []


def test_speed_cap_rejects_teleporting_detections():
    # Two points implying ~1000 km/h: must be rejected, not reported.
    pts = [TrackPoint(0, 0.0, 100, 300, 1, 0.9), TrackPoint(1, 1 / FPS, 900, 300, 1, 0.9)]
    try:
        estimate_speed(pts, CALIB)
    except ValueError:
        return
    raise AssertionError("impossible speed should raise, not report a number")


if __name__ == "__main__":
    test_extracts_serve_from_clutter()
    test_no_serve_in_pure_clutter()
    test_speed_cap_rejects_teleporting_detections()
    print("ALL TESTS PASS")
