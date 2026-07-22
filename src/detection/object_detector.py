"""Object recognition module.

Sole responsibility: recognize objects in a frame, and (separately) draw
bounding boxes and labels onto a frame. Knows nothing about where the frame
came from or how it will be displayed.

Detection and drawing are two separate operations on purpose: a caller can
run the expensive detection on one (possibly older) frame and then draw the
resulting boxes onto a different, newer frame -- which is what lets the
display stay smooth while detection runs at its own slower pace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


class ObjectDetector:
    """Wraps a YOLO model to recognize objects and annotate frames."""

    _TEXT_COLOR = (255, 255, 255)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.5

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        box_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold
        self._box_color = box_color

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Recognize objects in `frame` and return the list of detections."""
        results = self._model.predict(frame, conf=self._confidence_threshold, verbose=False)[0]

        return [
            Detection(
                label=self._model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=tuple(int(v) for v in box.xyxy[0]),
            )
            for box in results.boxes
        ]

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Return a copy of `frame` with the given detections drawn on it.

        `detections` need not have come from this exact frame -- drawing
        slightly-stale boxes onto a newer frame is expected and keeps the
        live video smooth while detection runs at its own pace.
        """
        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(annotated, detection)
        return annotated

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._box_color, 2)

        label = f"{detection.label} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, self._FONT, self._FONT_SCALE, 1)
        label_top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(frame, (x1, label_top), (x1 + text_w + 4, y1), self._box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2 if y1 - baseline - 2 > 0 else text_h),
            self._FONT,
            self._FONT_SCALE,
            self._TEXT_COLOR,
            1,
        )
