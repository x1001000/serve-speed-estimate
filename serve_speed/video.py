"""Video IO and annotation helpers (OpenCV based)."""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    if not info["fps"] or info["fps"] <= 0:
        info["fps"] = 30.0
    return info


def first_frame(path: str) -> np.ndarray:
    """Return the first frame as an RGB numpy array (for calibration UI)."""
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame from: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_frames(path: str, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, rgb_frame)`` for every ``stride``-th frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def _open_writer(path: str, fps: float, size: tuple[int, int]):
    """Open a VideoWriter, preferring an H.264/browser-friendly encoding."""
    for fourcc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not open a video writer with any known codec.")


def annotate_video(
    src_path: str,
    out_path: str,
    ball_boxes: dict[int, tuple[float, float, float, float]],
    trajectory: list[tuple[int, float, float]],
    calibration_pts: Optional[tuple[tuple[float, float], tuple[float, float]]],
    speed_text: str,
    max_width: int = 960,
) -> str:
    """Write an annotated copy of the video.

    ``ball_boxes`` maps ``frame_idx -> (x1, y1, x2, y2)`` for the tracked ball.
    ``trajectory`` is the ordered ``(frame_idx, x, y)`` ball path in source
    pixels.  All coordinates are in source resolution and are scaled down with
    the frame if it is wider than ``max_width``.
    """
    info = video_info(src_path)
    src_w, src_h = info["width"], info["height"]
    scale = min(1.0, max_width / src_w) if src_w else 1.0
    out_w, out_h = int(round(src_w * scale)), int(round(src_h * scale))
    fps = info["fps"]

    writer = _open_writer(out_path, fps, (out_w, out_h))

    def sp(pt):  # scale a point
        return int(round(pt[0] * scale)), int(round(pt[1] * scale))

    # Precompute trajectory polyline points keyed by how far along we are.
    traj_sorted = sorted(trajectory, key=lambda p: p[0])

    cap = cv2.VideoCapture(src_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            # Calibration reference line.
            if calibration_pts is not None:
                a, b = sp(calibration_pts[0]), sp(calibration_pts[1])
                cv2.line(frame, a, b, (0, 200, 255), 2)
                cv2.circle(frame, a, 5, (0, 200, 255), -1)
                cv2.circle(frame, b, 5, (0, 200, 255), -1)

            # Trajectory up to the current frame.
            drawn = [sp((x, y)) for (fi, x, y) in traj_sorted if fi <= idx]
            for p, q in zip(drawn, drawn[1:]):
                cv2.line(frame, p, q, (0, 255, 0), 2)
            for p in drawn[-30:]:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)

            # Current ball box.
            if idx in ball_boxes:
                x1, y1, x2, y2 = ball_boxes[idx]
                cv2.rectangle(frame, sp((x1, y1)), sp((x2, y2)), (0, 0, 255), 2)

            _draw_banner(frame, speed_text)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def _draw_banner(frame: np.ndarray, text: str) -> None:
    if not text:
        return
    lines = text.split("\n")
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    sizes = [cv2.getTextSize(ln, font, scale, thick)[0] for ln in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    h = sum(s[1] for s in sizes) + pad * (len(lines) + 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    y = pad
    for ln, (tw, th) in zip(lines, sizes):
        y += th + pad // 2
        cv2.putText(frame, ln, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
