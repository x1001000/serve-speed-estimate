"""Volleyball Serve Speed Estimator — Gradio app for Hugging Face Spaces.

Upload (or webcam-record) a video shot from the *side* of a volleyball court,
mark a segment of known real length (the 18 m court end-line to end-line, by
default), and the app detects and tracks the ball with RF-DETR + trackers and
estimates the serve speed from the trajectory length and time.
"""

from __future__ import annotations

# `spaces` must be imported before torch/CUDA is touched so that ZeroGPU can
# patch things correctly. RF-DETR/torch are only imported lazily deep inside
# the pipeline, so importing spaces first here is sufficient. When the package
# isn't installed (e.g. local dev off Hugging Face), fall back to a no-op
# decorator so the app still runs.
try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover - only when not on a Space

    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesShim()  # type: ignore

import logging
import tempfile

import cv2
import gradio as gr
import numpy as np

from serve_speed.detector import available_models
from serve_speed.pipeline import run_pipeline
from serve_speed.speed import Calibration
from serve_speed.video import annotate_video, first_frame

# Work around a gradio_client bug (present in the Gradio 5.9 line) where API
# schema generation crashes on boolean JSON sub-schemas (e.g.
# ``additionalProperties: true``) with "argument of type 'bool' is not
# iterable". That crash makes every request to "/" 500 and the browser report
# "No API found". Short-circuit boolean schemas so schema generation succeeds.
try:
    import gradio_client.utils as _gc_utils

    _orig_schema_to_type = _gc_utils._json_schema_to_python_type

    def _safe_schema_to_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_schema_to_type(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_schema_to_type
except Exception:  # pragma: no cover - never block startup on the patch
    pass

# ZeroGPU allocates a GPU only for the duration of a decorated call. Video
# detection/tracking is the GPU-heavy part, so we run just that under
# @spaces.GPU; annotation/plotting stay on CPU outside it.
#
# ``duration`` is the upfront budget requested from the ZeroGPU scheduler; the
# actual quota charged is the real runtime. A smaller budget means we can
# still fit under the remaining daily quota when it gets low, and the detector
# is cached across calls (see ``get_detector``) so warm calls finish well
# inside this window.
GPU_DURATION_S = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

DEFAULT_DISTANCE_M = 18.0  # volleyball court length, end line to end line

INTRO = """
# 🏐 Volleyball Serve Speed Estimator

Estimate a serve's speed from a **side-view** court video.

**How it works**
1. **Upload or record** a short clip of the serve (camera on the side, like a full-court view).
2. Click **Load frame for calibration**, then **click two points** on that frame that are a known real distance apart — by default the **18 m court length** (end line to end line along the near sideline).
3. Click **Estimate serve speed**.

The ball is found with [RF-DETR](https://github.com/roboflow/rf-detr) and linked
across frames with [trackers](https://github.com/roboflow/trackers); speed is the
trajectory length in metres divided by its time.

> ⚠️ This is an **estimate**. A single side-view scale can't fully correct for
> perspective/depth, so treat the number as a good ballpark, not a radar reading.
"""


def _draw_calibration(frame_rgb: np.ndarray, points: list) -> np.ndarray:
    img = frame_rgb.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, (255, 170, 0), -1)
        cv2.putText(
            img, str(i + 1), (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 170, 0), 2, cv2.LINE_AA,
        )
    if len(points) == 2:
        cv2.line(img, tuple(map(int, points[0])), tuple(map(int, points[1])), (255, 170, 0), 2)
    return img


def load_frame(video_path):
    if not video_path:
        raise gr.Error("Please upload or record a video first.")
    frame = first_frame(video_path)
    return frame, frame, [], "Frame loaded. Click the two ends of your known-distance line."


def on_click(frame_state, points, evt: gr.SelectData):
    if frame_state is None:
        raise gr.Error("Load a frame for calibration first.")
    x, y = evt.index[0], evt.index[1]
    if points is None or len(points) >= 2:
        points = [(x, y)]
    else:
        points = points + [(x, y)]
    display = _draw_calibration(frame_state, points)
    if len(points) == 1:
        status = "Point 1 set — now click the other end of the line."
    else:
        px = float(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))
        status = f"Calibration line set ({px:.0f}px). Adjust by clicking again, or estimate."
    return display, points, status


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


@spaces.GPU(duration=GPU_DURATION_S)
def _detect_and_track(video_path, p1, p2, distance_m, model_name, threshold, stride, tracker_name):
    """GPU-heavy stage: detect + track the ball and estimate the speed.

    This is the only part that needs a GPU, so it is the only part wrapped in
    ``@spaces.GPU``. ZeroGPU runs it in a separate process and serializes the
    boundary, so it takes only simple, picklable arguments and returns a
    ``PipelineResult`` (plain dataclasses / numpy arrays). Per-frame progress
    can't cross that boundary, so the outer handler shows coarse progress.
    """
    calibration = Calibration(
        p1=tuple(p1), p2=tuple(p2), real_distance_m=float(distance_m)
    )
    return run_pipeline(
        video_path,
        calibration=calibration,
        model_name=model_name,
        threshold=float(threshold),
        stride=int(stride),
        tracker_name=tracker_name,
        progress=None,
    )


def estimate(video_path, frame_state, points, distance_m, model_name, threshold, stride, tracker_name, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload or record a video first.")
    if not points or len(points) != 2:
        raise gr.Error("Mark exactly two calibration points on the loaded frame.")
    if not distance_m or distance_m <= 0:
        raise gr.Error("Enter a positive real distance for the calibration line.")

    calibration = Calibration(p1=points[0], p2=points[1], real_distance_m=float(distance_m))

    progress(0.05, desc="Allocating GPU · detecting & tracking ball…")
    try:
        result = _detect_and_track(
            video_path,
            tuple(points[0]),
            tuple(points[1]),
            float(distance_m),
            model_name,
            float(threshold),
            int(stride),
            tracker_name,
        )
    except ValueError as exc:
        raise gr.Error(str(exc))

    est = result.estimate
    speed_text = (
        f"Serve speed (peak): {est.peak_kmh:.1f} km/h\n"
        f"Average (flight): {est.avg_kmh:.1f} km/h"
    )

    progress(0.97, desc="Rendering annotated video…")
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    traj = [(p.frame_idx, p.x, p.y) for p in result.trajectory]
    annotate_video(
        video_path,
        out_path,
        ball_boxes=result.ball_boxes,
        trajectory=traj,
        calibration_pts=(points[0], points[1]),
        speed_text=speed_text,
    )

    report = (
        f"## Estimated serve speed\n\n"
        f"### 🏐 {est.peak_kmh:.1f} km/h  ·  {est.peak_ms:.1f} m/s  _(peak)_\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Peak speed | **{est.peak_kmh:.1f} km/h** ({est.peak_ms:.1f} m/s) |\n"
        f"| Average over flight | {est.avg_kmh:.1f} km/h ({est.avg_ms:.1f} m/s) |\n"
        f"| Trajectory length | {est.path_length_m:.2f} m |\n"
        f"| Flight time (tracked) | {est.duration_s:.2f} s |\n"
        f"| Ball detections used | {est.n_points} |\n"
        f"| Scale | {est.meters_per_pixel*100:.2f} cm/px "
        f"(from {distance_m:.1f} m over {calibration.pixel_distance:.0f} px) |\n\n"
        f"_Peak speed is the fastest part of the flight — closest to the speed just "
        f"after contact._"
    )

    return out_path, report, _speed_plot(est)


with gr.Blocks(title="Volleyball Serve Speed Estimator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    frame_state = gr.State(None)
    points_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(
                label="Serve video (upload or record)",
                sources=["upload", "webcam"],
                include_audio=False,
            )
            gr.Examples(
                examples=[
                    ["examples/PXL_20260717_054002690_h264.mp4"],
                    ["examples/POL.mp4"],
                ],
                inputs=[video_in],
                label="Example serves (click to load, then calibrate)",
            )
            load_btn = gr.Button("① Load frame for calibration", variant="secondary")
            calib_image = gr.Image(
                label="② Click two ends of a known-distance line",
                interactive=False,
                type="numpy",
            )
            status = gr.Markdown("Upload a video and load a frame to begin.")
            distance = gr.Number(
                value=DEFAULT_DISTANCE_M,
                label="Real distance of the marked line (m)",
                info="Court length end-to-end is 18 m. Use any known reference you can see.",
            )

        with gr.Column(scale=1):
            with gr.Accordion("Detection settings", open=False):
                model_dd = gr.Dropdown(
                    choices=available_models(), value="large", label="RF-DETR model",
                    info="Larger = more accurate on a fast ball. On ZeroGPU even 'large' is fast.",
                )
                threshold = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Detection confidence")
                stride = gr.Slider(1, 5, value=1, step=1, label="Frame stride",
                                   info="Process every Nth frame. 1 = best time resolution.")
                tracker_dd = gr.Dropdown(
                    choices=["sort", "bytetrack", "ocsort"], value="bytetrack", label="Tracker",
                )
            estimate_btn = gr.Button("③ Estimate serve speed", variant="primary")
            report_md = gr.Markdown()
            speed_fig = gr.Plot(label="Speed over trajectory")
            video_out = gr.Video(label="Annotated trajectory")

    load_btn.click(load_frame, inputs=[video_in], outputs=[calib_image, frame_state, points_state, status])
    calib_image.select(on_click, inputs=[frame_state, points_state], outputs=[calib_image, points_state, status])
    estimate_btn.click(
        estimate,
        inputs=[video_in, frame_state, points_state, distance, model_dd, threshold, stride, tracker_dd],
        outputs=[video_out, report_md, speed_fig],
    )


if __name__ == "__main__":
    demo.launch()
