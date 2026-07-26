"""Volleyball serve-speed estimation from a side-view court video.

The pipeline is intentionally small and composable:

- ``detector``  -> RF-DETR wrapper that finds the ball in a single frame.
- ``tracking``  -> links per-frame ball detections into a trajectory with
  the ``trackers`` library.
- ``speed``     -> turns a pixel-space trajectory + a calibration line into
  a real-world speed estimate.
- ``video``     -> frame IO and annotation helpers.
"""

from .speed import Calibration, SpeedEstimate, estimate_speed

__all__ = ["Calibration", "SpeedEstimate", "estimate_speed"]
