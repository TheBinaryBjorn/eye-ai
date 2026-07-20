"""Output stream module.

Sole responsibility: composite the annotated frame and the object
descriptions into a single window, and report key presses from that window
(quit, voice trigger, etc). Knows nothing about how frames are captured,
detected, or described, and nothing about what a given key press means.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.description.object_describer import ObjectDescription


class StreamDisplay:
    """Shows the annotated frame plus a side panel listing detections."""

    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _HEADING_SCALE = 0.6
    _TEXT_SCALE = 0.45
    _LINE_HEIGHT = 18
    _MARGIN = 16

    def __init__(
        self,
        window_name: str,
        panel_width: int,
        panel_bg_color: Tuple[int, int, int],
        heading_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        subtext_color: Tuple[int, int, int],
    ) -> None:
        self._window_name = window_name
        self._panel_width = panel_width
        self._panel_bg_color = panel_bg_color
        self._heading_color = heading_color
        self._text_color = text_color
        self._subtext_color = subtext_color
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def show(self, annotated_frame: np.ndarray, descriptions: List[ObjectDescription]) -> None:
        panel = self._render_panel(annotated_frame.shape[0], descriptions)
        canvas = np.hstack([annotated_frame, panel])
        cv2.imshow(self._window_name, canvas)

    def poll_key(self) -> Optional[str]:
        """Return the character key pressed since the last poll, if any."""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return None
        return chr(key)

    def close(self) -> None:
        cv2.destroyAllWindows()

    def __enter__(self) -> "StreamDisplay":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _render_panel(self, height: int, descriptions: List[ObjectDescription]) -> np.ndarray:
        panel = np.full((height, self._panel_width, 3), self._panel_bg_color, dtype=np.uint8)
        text_width = self._panel_width - 2 * self._MARGIN

        y = self._MARGIN + self._LINE_HEIGHT
        cv2.putText(
            panel, "Detected Objects", (self._MARGIN, y),
            self._FONT, self._HEADING_SCALE, self._heading_color, 2,
        )
        y += int(self._LINE_HEIGHT * 1.5)

        if not descriptions:
            cv2.putText(
                panel, "No objects detected", (self._MARGIN, y),
                self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
            )
            return panel

        for index, description in enumerate(descriptions, start=1):
            heading = f"{index}. {description.label.capitalize()} ({description.confidence:.0%})"
            body = (
                f"{description.size_label}, {description.width_px}x{description.height_px}px, "
                f"predominantly {description.color_name}. {description.feature_text}"
            )
            lines = [heading] + self._wrap_text(body, self._TEXT_SCALE, text_width)

            if y + len(lines) * self._LINE_HEIGHT > height - self._MARGIN:
                remaining = len(descriptions) - index + 1
                cv2.putText(
                    panel, f"+ {remaining} more (resize window)", (self._MARGIN, y),
                    self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
                )
                break

            cv2.putText(panel, lines[0], (self._MARGIN, y), self._FONT, self._TEXT_SCALE, self._text_color, 1)
            y += self._LINE_HEIGHT
            for line in lines[1:]:
                cv2.putText(panel, line, (self._MARGIN, y), self._FONT, self._TEXT_SCALE, self._subtext_color, 1)
                y += self._LINE_HEIGHT
            y += self._LINE_HEIGHT // 2

        return panel

    def _wrap_text(self, text: str, scale: float, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            (text_w, _), _ = cv2.getTextSize(candidate, self._FONT, scale, 1)
            if text_w <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
