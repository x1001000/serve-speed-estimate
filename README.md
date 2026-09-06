---
title: Volleyball Serve Speed Estimator
emoji: 🏐
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
license: mit
short_description: Estimate volleyball serve speed from a side-view court video
---

# 🏐 Volleyball Serve Speed Estimator

A Gradio app (built for [Hugging Face Spaces](https://huggingface.co/spaces))
that estimates a **volleyball serve's speed** from a video shot at the **side of
the court** — the kind of full-court view you'd get filming from the sideline.

You **click the ball** once while it's in flight; a **click-seeded motion
tracker** follows it forwards and backwards along its arc using frame-difference
motion detection. Speed is then the trajectory's real-world length divided by
its elapsed time. It runs on **CPU** — no GPU, no model download.

### Why click-to-track instead of an object detector?

In a wide full-court shot the ball can be only **~7 px across** and
motion-blurred. A generic object detector (e.g. COCO "sports ball") misses it
and fires on everything else round — players, a ball on the floor, wall pads —
producing nonsense speeds. But a small fast ball *is* a bright **moving** blob,
which frame-differencing spots reliably. The only ambiguity is *which* blob is
the ball among all the moving people, and your single click resolves exactly
that. A projectile motion model then carries the track across large per-frame
jumps.

## How it works

1. **Upload or webcam-record** a short clip of the serve.
2. **Calibrate:** on the calibration frame, **click two points** a known real
   distance apart. The default reference is the **18 m court length** (end line
   to end line), but any visible known distance works (net height 2.43 m / 2.24 m,
   a 9 m sideline, etc.).
3. **Mark the ball:** drag the **serve-frame slider** to a moment when the ball is
   *in flight* and **click on the ball**. You don't need to be precise or catch the
   exact contact frame — clicking anywhere on the visible flight works (the tracker
   snaps to the nearest moving blob and walks the arc both directions).
4. **Estimate** — you get **peak** and **average flight** speed (km/h + m/s), a
   **speed-over-time plot**, and an **annotated video** of the tracked trajectory.

## The estimation model

The ball is followed by a **click-seeded motion-ballistic tracker**:

- **Motion blobs** per frame come from three-frame differencing
  (`|cur−prev| AND |next−cur|`) — a tiny fast ball is a bright *moving* blob.
- Starting from your click, the tracker follows the blob nearest the **predicted**
  next position (constant-velocity + gravity), so large per-frame jumps are fine.
- It stops at end-of-flight (direction reversal / lost / impossible jump) and
  trims the result to its clean **projectile** span (x ≈ linear in time,
  y ≈ quadratic).

Then, for the tracked centres and a calibration scale
`metres_per_pixel = known_distance / pixel_distance`:

```
segment_speed_i = distance(p_i, p_{i+1}) * metres_per_pixel / (t_{i+1} - t_i)
peak_speed      = max( median_filtered(segment_speed) )   # capped at 45 m/s
```

### Accuracy caveats

This uses a **single scalar scale**, exact only for motion in the plane of the
calibration line; a real serve arcs in 3D and side-view perspective shifts the
ball off that plane. For the best results:

- **Click the ball while it's clearly in flight** (mid-arc is ideal).
- Prefer a **higher frame rate** (60–120 fps): less blur, more points.
- Calibrate along the plane the ball actually travels in (near sideline).
- Treat the number as a **good estimate**, not a radar-gun measurement.

## Hardware

Runs on **CPU basic** (free) — frame-difference tracking is light; no GPU or
model download.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open the printed local URL.

## Project layout

```
app.py                  Gradio UI (calibrate + click-the-ball)
serve_speed/
  motion_tracker.py     click-seeded frame-difference ball tracker
  tracking.py           TrackPoint type
  speed.py              calibration + speed math
  video.py              frame IO and annotation
  pipeline.py           click + video -> trajectory -> speed estimate
tests/                  synthetic-video tracker tests
```

## Tests

```bash
PYTHONPATH=. python tests/test_tracker.py
```

## License

MIT
