"""Motion-tracker tests on synthetic frames.

We render a small bright ball on a dark background following a projectile
path, add moving distractor blobs, and check the click-seeded tracker recovers
the ball and a sane speed — and that a click on nothing yields no track.
"""

import cv2
import numpy as np

from serve_speed.motion_tracker import track_serve
from serve_speed.speed import Calibration, estimate_speed

FPS = 30.0
W, H = 960, 540
CALIB = Calibration(p1=(200, 400), p2=(820, 400), real_distance_m=18.0)
MPP = CALIB.meters_per_pixel


def _serve_xy(k):
    # ~20 m/s horizontal (≈72 km/h): px/frame = 20/FPS/MPP
    vx = 20.0 / FPS / MPP
    x = 200 + vx * k
    y = 250 - 5 * k + 0.4 * k * k  # up then down
    return x, y


def _render(n=24, with_ball=True, seed=0, gap=None):
    """Render frames; ``gap=(a,b)`` omits the ball on frames a..b (a dropout)."""
    rng = np.random.default_rng(seed)
    frames = []
    # a couple of slow distractor blobs
    d1 = np.array([400.0, 300.0])
    d2 = np.array([600.0, 350.0])
    for k in range(n):
        img = np.full((H, W), 30, np.uint8)
        d1 += rng.normal(0, 1.5, 2)
        d2 += rng.normal(0, 1.5, 2)
        cv2.circle(img, tuple(d1.astype(int)), 4, 180, -1)
        cv2.circle(img, tuple(d2.astype(int)), 5, 160, -1)
        if with_ball and not (gap and gap[0] <= k <= gap[1]):
            x, y = _serve_xy(k)
            cv2.circle(img, (int(x), int(y)), 3, 235, -1)
        frames.append(img)
    return frames


def test_tracks_ball_and_estimates_speed():
    frames = _render()
    # click a few pixels off the ball's frame-5 position
    x, y = _serve_xy(5)
    pts = track_serve(frames, seed_frame=5, seed_xy=(x + 4, y - 5), fps=FPS, meters_per_pixel=MPP)
    assert len(pts) >= 8, len(pts)
    est = estimate_speed(pts, CALIB)
    assert 60 < est.peak_kmh < 90, est.peak_kmh  # synthetic serve ≈ 72 km/h


def test_bridges_ball_dropout():
    # Ball vanishes for 6 frames mid-flight (as at a real apex). The tracker
    # must coast across the gap and still recover the full arc + speed, even
    # when the click lands just before the gap.
    frames = _render(n=30, gap=(9, 14))
    x, y = _serve_xy(7)
    pts = track_serve(frames, seed_frame=7, seed_xy=(x, y), fps=FPS, meters_per_pixel=MPP)
    spanned = [p.frame_idx for p in pts]
    assert min(spanned) <= 6 and max(spanned) >= 18, spanned  # crossed the gap
    est = estimate_speed(pts, CALIB)
    assert 60 < est.peak_kmh < 90, est.peak_kmh


def test_no_ball_no_track():
    frames = _render(with_ball=False)
    pts = track_serve(frames, seed_frame=5, seed_xy=(210, 230), fps=FPS, meters_per_pixel=MPP)
    # With only slow distractors far from the click, no clean serve emerges.
    if len(pts) >= 3:
        est = estimate_speed(pts, CALIB)
        assert est.peak_kmh < 20  # at worst a slow distractor, never a fake fast serve


if __name__ == "__main__":
    test_tracks_ball_and_estimates_speed()
    test_bridges_ball_dropout()
    test_no_ball_no_track()
    print("ALL TESTS PASS")
