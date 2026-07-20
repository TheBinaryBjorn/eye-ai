"""Object recognition module.

Sole responsibility: given a frame, recognize objects, draw bounding boxes
and labels, and return the annotated frame. Knows nothing about where the
frame came from or how it will be displayed.
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
    """Wraps a YOLO model to recognize and annotate objects in frames."""

    _BOX_COLOR = (0, 255, 0)
    _TEXT_COLOR = (0, 0, 0)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.5

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5) -> None:
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """Recognize objects in `frame` and return (annotated_frame, detections)."""
        results = self._model.predict(frame, conf=self._confidence_threshold, verbose=False)[0]

        detections = [
            Detection(
                label=self._model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=tuple(int(v) for v in box.xyxy[0]),
            )
            for box in results.boxes
        ]

        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(annotated, detection)
        return annotated, detections

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._BOX_COLOR, 2)

        label = f"{detection.label} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, self._FONT, self._FONT_SCALE, 1)
        label_top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(frame, (x1, label_top), (x1 + text_w + 4, y1), self._BOX_COLOR, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2 if y1 - baseline - 2 > 0 else text_h),
            self._FONT,
            self._FONT_SCALE,
            self._TEXT_COLOR,
            1,
        )
