"""Volleyball Serve Speed Estimator — Gradio app for Hugging Face Spaces.

Workflow:
1. Upload or record a side-view clip of the serve.
2. Calibrate: click two points a known distance apart (default 18 m court length).
3. Scrub to a frame where the ball is in flight and click on the ball.
4. A click-seeded motion tracker follows the ball and reports the serve speed.

The ball is followed by frame-difference motion tracking (CPU, no model
download), which handles a tiny/blurred ball that appearance detectors miss.
"""

from __future__ import annotations

import logging
import tempfile

import cv2
import gradio as gr
import numpy as np

from serve_speed.pipeline import run_pipeline
from serve_speed.speed import Calibration
from serve_speed.video import annotate_video, first_frame, frame_at, video_info

# Work around a gradio_client bug (Gradio 5.9 line) where API schema generation
# crashes on boolean JSON sub-schemas ("argument of type 'bool' is not
# iterable"), 500-ing every request and showing "No API found" in the browser.
try:
    import gradio_client.utils as _gc_utils

    _orig_schema_to_type = _gc_utils._json_schema_to_python_type

    def _safe_schema_to_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_schema_to_type(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_schema_to_type
except Exception:  # pragma: no cover
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

DEFAULT_DISTANCE_M = 18.0  # volleyball court length, end line to end line

INTRO = """
# 🏐 Volleyball Serve Speed Estimator

Estimate a serve's speed from a **side-view** court video — runs on CPU, no model download.

**How it works**
1. **Upload or record** a short clip of the serve (camera on the side).
2. **Calibrate:** on the calibration frame, click the **two ends of a known distance** — by default the **18 m court length** (end line to end line).
3. **Mark the ball:** drag the **serve-frame slider** to a moment when the ball is *in flight*, then **click on the ball**. You don't need to be precise or catch the exact contact frame — clicking anywhere on the visible flight works.
4. Click **Estimate serve speed**.

A motion tracker follows the ball forwards and backwards from your click along its
flight path; speed is the trajectory length in metres divided by its time.

> ⚠️ This is an **estimate**. A single side-view scale can't fully correct for
> perspective/depth, so treat it as a good ballpark, not a radar reading.
"""


def _draw_markers(frame_rgb, points, color, connect=False):
    img = frame_rgb.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, color, 2)
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
        cv2.putText(img, str(i + 1), (int(x) + 11, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if connect and len(points) == 2:
        cv2.line(img, tuple(map(int, points[0])), tuple(map(int, points[1])), color, 2)
    return img


def init_video(video_path):
    """When a video is set, load the calibration frame and the seed frame."""
    if not video_path:
        return (None, None, [], "Upload or record a video to begin.",
                gr.update(maximum=1, value=0), None, None, None,
                "Load a video, then scrub to the ball in flight.")
    info = video_info(video_path)
    n = max(1, info["frame_count"])
    calib = first_frame(video_path)
    # default the serve slider to the middle of the clip (serves are usually mid-clip)
    mid = min(n - 1, n // 3)
    serve = frame_at(video_path, mid)
    return (
        calib, calib, [], "Calibration frame loaded — click the two ends of your known-distance line.",
        gr.update(maximum=max(1, n - 1), value=mid, step=1),
        serve, serve, None,
        "Scrub to a frame where the ball is in flight, then click on the ball.",
    )


def on_calib_click(calib_frame, points, evt: gr.SelectData):
    if calib_frame is None:
        raise gr.Error("Load a video first.")
    x, y = evt.index[0], evt.index[1]
    if points is None or len(points) >= 2:
        points = [(x, y)]
    else:
        points = points + [(x, y)]
    display = _draw_markers(calib_frame, points, (255, 170, 0), connect=True)
    if len(points) == 1:
        status = "Point 1 set — click the other end of the line."
    else:
        px = float(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))
        status = f"Calibration line set ({px:.0f}px). Click again to redo."
    return display, points, status


def on_serve_slider(video_path, idx):
    if not video_path:
        return None, None, None, "Load a video first."
    frame = frame_at(video_path, int(idx))
    return frame, frame, None, f"Frame {int(idx)} — click on the ball if it's in flight here."


def on_seed_click(serve_frame, serve_idx, evt: gr.SelectData):
    if serve_frame is None:
        raise gr.Error("Load a video and pick a serve frame first.")
    x, y = evt.index[0], evt.index[1]
    seed = (int(serve_idx), float(x), float(y))
    display = _draw_markers(serve_frame, [(x, y)], (0, 255, 0))
    return display, seed, f"Ball marked at frame {int(serve_idx)} ({x}, {y}). Ready to estimate."


def _speed_plot(estimate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    ax.plot(estimate.times, estimate.speeds_ms * 3.6, marker="o", ms=3, color="#1f77b4")
    ax.axhline(estimate.peak_kmh, color="#d62728", ls="--", lw=1, label=f"peak {estimate.peak_kmh:.1f} km/h")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title("Ball speed along trajectory")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def estimate(video_path, calib_points, distance_m, seed, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload or record a video first.")
    if not calib_points or len(calib_points) != 2:
        raise gr.Error("Mark the two calibration points on the calibration frame.")
    if not distance_m or distance_m <= 0:
        raise gr.Error("Enter a positive real distance for the calibration line.")
    if not seed:
        raise gr.Error("Click on the ball in a frame where it is in flight.")

    calibration = Calibration(p1=calib_points[0], p2=calib_points[1], real_distance_m=float(distance_m))
    seed_frame, sx, sy = seed

    def _progress(frac, desc):
        progress(frac, desc=desc)

    try:
        result = run_pipeline(
            video_path, calibration=calibration,
            seed_frame=seed_frame, seed_xy=(sx, sy), progress=_progress,
        )
    except ValueError as exc:
        raise gr.Error(str(exc))

    est = result.estimate
    speed_text = (
        f"Serve speed (peak): {est.peak_kmh:.1f} km/h\n"
        f"Average (flight): {est.avg_kmh:.1f} km/h"
    )

    progress(0.9, desc="Rendering annotated video…")
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    traj = [(p.frame_idx, p.x, p.y) for p in result.trajectory]
    annotate_video(
        video_path, out_path,
        ball_boxes=result.ball_boxes, trajectory=traj,
        calibration_pts=(calib_points[0], calib_points[1]), speed_text=speed_text,
    )

    report = (
        f"## Estimated serve speed\n\n"
        f"### 🏐 {est.peak_kmh:.1f} km/h  ·  {est.peak_ms:.1f} m/s  _(peak)_\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Peak speed | **{est.peak_kmh:.1f} km/h** ({est.peak_ms:.1f} m/s) |\n"
        f"| Average over flight | {est.avg_kmh:.1f} km/h ({est.avg_ms:.1f} m/s) |\n"
        f"| Trajectory length | {est.path_length_m:.2f} m |\n"
        f"| Flight time (tracked) | {est.duration_s:.2f} s |\n"
        f"| Ball positions tracked | {est.n_points} |\n"
        f"| Scale | {est.meters_per_pixel*100:.2f} cm/px |\n\n"
        f"_Peak speed is the fastest part of the flight — closest to the speed just "
        f"after contact._"
    )
    return out_path, report, _speed_plot(est)


with gr.Blocks(title="Volleyball Serve Speed Estimator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    calib_frame_state = gr.State(None)
    calib_points_state = gr.State([])
    serve_frame_state = gr.State(None)
    seed_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(
                label="Serve video (upload or record)",
                sources=["upload", "webcam"], include_audio=False,
            )
            gr.Examples(
                examples=[["examples/PXL_20260717_054002690_h264.mp4"]],
                inputs=[video_in],
                label="Example serve (click to load, then calibrate & mark the ball)",
            )
            gr.Markdown("### ① Calibrate — click the two ends of a known distance")
            calib_image = gr.Image(label="Calibration frame", interactive=False, type="numpy")
            calib_status = gr.Markdown("Upload or record a video to begin.")
            distance = gr.Number(
                value=DEFAULT_DISTANCE_M, label="Real distance of the marked line (m)",
                info="Court length end-to-end is 18 m. Any known visible distance works.",
            )

        with gr.Column(scale=1):
            gr.Markdown("### ② Mark the ball — scrub to it in flight, then click it")
            serve_slider = gr.Slider(0, 1, value=0, step=1, label="Serve frame")
            serve_image = gr.Image(label="Click on the ball", interactive=False, type="numpy")
            seed_status = gr.Markdown("Load a video, then scrub to the ball in flight.")
            estimate_btn = gr.Button("③ Estimate serve speed", variant="primary")
            report_md = gr.Markdown()
            speed_fig = gr.Plot(label="Speed over trajectory")
            video_out = gr.Video(label="Annotated trajectory")

    video_in.change(
        init_video, inputs=[video_in],
        outputs=[calib_image, calib_frame_state, calib_points_state, calib_status,
                 serve_slider, serve_image, serve_frame_state, seed_state, seed_status],
    )
    calib_image.select(on_calib_click, inputs=[calib_frame_state, calib_points_state],
                       outputs=[calib_image, calib_points_state, calib_status])
    serve_slider.change(on_serve_slider, inputs=[video_in, serve_slider],
                        outputs=[serve_image, serve_frame_state, seed_state, seed_status])
    serve_image.select(on_seed_click, inputs=[serve_frame_state, serve_slider],
                       outputs=[serve_image, seed_state, seed_status])
    estimate_btn.click(
        estimate,
        inputs=[video_in, calib_points_state, distance, seed_state],
        outputs=[video_out, report_md, speed_fig],
    )


if __name__ == "__main__":
    demo.launch()
