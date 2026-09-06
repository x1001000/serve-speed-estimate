"""Shared trajectory point type.

The project previously linked appearance detections with the ``trackers``
package; it now follows the ball with a click-seeded motion tracker
(``motion_tracker``). ``TrackPoint`` remains the common currency between the
tracker and the speed math.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackPoint:
    """One observation of the ball along the trajectory."""

    frame_idx: int
    t: float  # seconds
    x: float  # pixel x of ball centre
    y: float  # pixel y of ball centre
    tracker_id: int = 1
    conf: float = 1.0
