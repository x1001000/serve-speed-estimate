"""RF-DETR based ball detector.

RF-DETR (https://github.com/roboflow/rf-detr) ships COCO-pretrained
checkpoints.  A volleyball is detected as the COCO ``sports ball`` class, so
we resolve that class id from the package's own class map at runtime instead
of hard-coding a magic number (the id differs between the 80- and 91-class
COCO conventions).
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model name -> RF-DETR class.  Smaller = faster (better for the free CPU
# tier on Hugging Face Spaces), larger = more accurate on a small, fast ball.
_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Names we accept as "the ball".  Kept as an exact set so we don't accidentally
# match e.g. "baseball bat".
_BALL_NAMES = {"sports ball", "sports_ball", "ball", "volleyball"}

# Fallback ids covering both common COCO conventions ("sports ball" is 32 in
# the 80-class list, 37 in the 91-class list) in case the class map cannot be
# imported for some reason.
_FALLBACK_BALL_IDS = {32, 37}


def available_models() -> list[str]:
    return list(_MODEL_CLASSES.keys())


@lru_cache(maxsize=1)
def get_detector(model_name: str = "large", resolution: Optional[int] = None) -> "BallDetector":
    """Return a process-wide cached detector for ``model_name``.

    ZeroGPU keeps the worker process warm between ``@spaces.GPU`` calls, so
    caching here means the RF-DETR weights are loaded (and moved to GPU) only
    on the first call. Later calls reuse the same instance and skip the
    multi-second warmup that would otherwise burn ZeroGPU quota. The cache
    holds a single entry so switching model size evicts the previous weights
    from GPU memory instead of pinning both.
    """
    return BallDetector(model_name=model_name, resolution=resolution)


@lru_cache(maxsize=1)
def _coco_classes() -> dict[int, str]:
    """Return an ``{id: name}`` map from whichever rfdetr module exposes it."""
    for path in (
        "rfdetr.util.coco_classes",
        "rfdetr.assets.coco_classes",
        "rfdetr.detr.util.coco_classes",
    ):
        try:
            mod = importlib.import_module(path)
        except Exception:  # pragma: no cover - depends on installed version
            continue
        classes = getattr(mod, "COCO_CLASSES", None)
        if classes is None:
            continue
        if isinstance(classes, dict):
            return {int(k): str(v) for k, v in classes.items()}
        if isinstance(classes, (list, tuple)):
            return {i: str(name) for i, name in enumerate(classes)}
    logger.warning("Could not import RF-DETR COCO class map; using fallback ids.")
    return {}


def ball_class_ids() -> set[int]:
    """Resolve the set of class ids that count as a volleyball."""
    classes = _coco_classes()
    ids = {cid for cid, name in classes.items() if name.strip().lower() in _BALL_NAMES}
    return ids or set(_FALLBACK_BALL_IDS)


class BallDetector:
    """Thin wrapper around an RF-DETR checkpoint that returns ball detections.

    Detection output is a plain numpy array of ``[x1, y1, x2, y2, confidence]``
    rows so the rest of the pipeline has no hard dependency on a particular
    ``supervision`` version.  The original ``supervision.Detections`` is also
    returned for the tracker, which speaks that format natively.
    """

    def __init__(self, model_name: str = "large", resolution: Optional[int] = None):
        if model_name not in _MODEL_CLASSES:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose one of {available_models()}."
            )
        self.model_name = model_name
        self._resolution = resolution
        self._model = None
        self._ball_ids = ball_class_ids()

    def _load(self):
        if self._model is None:
            from rfdetr import __dict__ as rfdetr_ns  # lazy: heavy import

            cls = rfdetr_ns.get(_MODEL_CLASSES[self.model_name])
            if cls is None:  # pragma: no cover - version mismatch guard
                from rfdetr import RFDETRMedium as cls  # type: ignore
            kwargs = {}
            if self._resolution:
                kwargs["resolution"] = self._resolution
            logger.info("Loading RF-DETR (%s)…", self.model_name)
            self._model = cls(**kwargs)
            # optimize_for_inference is available on recent versions and gives
            # a noticeable speed-up; ignore if unsupported.
            try:
                self._model.optimize_for_inference()
            except Exception:  # pragma: no cover
                pass
        return self._model

    def detect(self, frame_rgb: np.ndarray, threshold: float = 0.4):
        """Detect balls in an RGB frame.

        Returns ``(boxes, detections)`` where ``boxes`` is an ``(N, 5)`` array
        of ``[x1, y1, x2, y2, conf]`` for ball detections only, sorted by
        confidence (highest first), and ``detections`` is the filtered
        ``supervision.Detections`` object for the tracker.
        """
        model = self._load()
        # RF-DETR's predict is most reliable with a PIL RGB image.
        from PIL import Image

        image = Image.fromarray(np.asarray(frame_rgb).astype(np.uint8))
        detections = model.predict(image, threshold=threshold)

        class_id = getattr(detections, "class_id", None)
        if class_id is not None and len(class_id) > 0:
            mask = np.isin(np.asarray(class_id), list(self._ball_ids))
            detections = detections[mask]

        xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
        conf = getattr(detections, "confidence", None)
        if conf is None or len(conf) == 0:
            conf = np.ones(len(xyxy))
        conf = np.asarray(conf, dtype=float)

        if len(xyxy) == 0:
            return np.empty((0, 5), dtype=float), detections

        order = np.argsort(-conf)
        boxes = np.column_stack([xyxy, conf])[order]
        return boxes, detections[order] if len(xyxy) else detections
