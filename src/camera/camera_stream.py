"""Camera hardware interaction module.

Sole responsibility: open a webcam device and hand back raw frames.
Knows nothing about detection or display.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Owns the webcam device handle and produces raw frames."""

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera at index {camera_index}")

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self) -> Optional[np.ndarray]:
        """Return the next frame, or None if the stream has ended/failed."""
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()
