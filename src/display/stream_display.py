"""Output stream module.

Sole responsibility: display the raw input stream alongside the annotated
output stream, and report when the user wants to quit. Knows nothing about
how frames are captured or how objects are recognized.
"""

from __future__ import annotations

import cv2
import numpy as np


class StreamDisplay:
    """Shows the input and output streams in separate windows."""

    def __init__(self, input_window_name: str, output_window_name: str) -> None:
        self._input_window_name = input_window_name
        self._output_window_name = output_window_name
        cv2.namedWindow(self._input_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self._output_window_name, cv2.WINDOW_NORMAL)

    def show(self, input_frame: np.ndarray, output_frame: np.ndarray) -> None:
        cv2.imshow(self._input_window_name, input_frame)
        cv2.imshow(self._output_window_name, output_frame)

    def should_quit(self, quit_key: str = "q") -> bool:
        return cv2.waitKey(1) & 0xFF == ord(quit_key)

    def close(self) -> None:
        cv2.destroyAllWindows()

    def __enter__(self) -> "StreamDisplay":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
