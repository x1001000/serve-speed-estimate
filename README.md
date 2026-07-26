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
suggested_hardware: zero-a10g
short_description: Estimate volleyball serve speed from a side-view court video
---

# 🏐 Volleyball Serve Speed Estimator

A Gradio app (built for [Hugging Face Spaces](https://huggingface.co/spaces))
that estimates a **volleyball serve's speed** from a video shot at the **side of
the court** — the kind of full-court view you'd get filming from the sideline.

The ball is detected with [RF-DETR](https://github.com/roboflow/rf-detr) and
linked across frames into a trajectory with the
[trackers](https://github.com/roboflow/trackers) library. Speed is then the
trajectory's real-world length divided by its elapsed time.

## How it works

1. **Upload or webcam-record** a short clip of the serve.
2. **Load a frame** and **click two points** a known real distance apart to
   calibrate pixels → metres. The default reference is the **18 m court length**
   (end line to end line along the near sideline), but any visible known
   distance works (net height 2.43 m / 2.24 m, a 9 m sideline, etc.).
3. **Estimate** — you get:
   - **Peak speed** (fastest part of the flight, closest to the speed just after
     contact) and **average flight speed**, in km/h and m/s,
   - a **speed-over-time plot**, and
   - an **annotated video** showing the ball box, its trajectory, and the
     calibration line.

## The estimation model

For a ball centre observed at pixel positions over time, and a calibration
scale `metres_per_pixel = known_distance / pixel_distance`:

```
segment_speed_i = distance(p_i, p_{i+1}) * metres_per_pixel / (t_{i+1} - t_i)
peak_speed      = max( median_filtered(segment_speed) )
```

The trajectory is smoothed and outlier-filtered before the peak is taken.

### Accuracy caveats

This uses a **single scalar scale**, which is exact only for motion in the plane
of the calibration line. A real serve arcs in 3D, and side-view perspective
means the ball is nearer/farther than that plane at different times. So:

- Calibrate along the plane the ball actually travels in (near sideline) for the
  best result.
- Treat the number as a **good estimate**, not a radar-gun measurement.
- Higher frame-rate clips and a larger RF-DETR model improve the estimate.

## Hardware / ZeroGPU

The GPU-heavy stage (ball detection + tracking) is wrapped in `@spaces.GPU`, so
this Space runs well on **ZeroGPU** (`zero-a10g`), which needs a Hugging Face
PRO account. Only that call allocates a GPU; annotation and plotting stay on
CPU. Off a Space (local dev), `spaces` isn't required — the decorator falls back
to a no-op and the app runs on whatever device torch finds.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open the printed local URL. First run downloads the RF-DETR checkpoint.

## Project layout

```
app.py                  Gradio UI
serve_speed/
  detector.py           RF-DETR ball detection (resolves the COCO ball class)
  tracking.py           trackers-based linking + serve-trajectory selection
  speed.py              calibration + speed math
  video.py              frame IO and annotation
  pipeline.py           video -> trajectory -> speed estimate
```

## License

MIT
