"""Output stream module.

Sole responsibility: composite the annotated frame, the object descriptions,
the voice pipeline's current phase, and the chat history into a single
window, and report key presses from that window (quit, voice trigger,
etc). Knows nothing about how frames are captured, detected, described, or
how the voice pipeline actually works -- it only renders the plain data
(and state label) it's handed.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.description.object_describer import ObjectDescription
from src.llm.conversational_agent import ChatMessage


class VoiceState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class StreamDisplay:
    """Shows the annotated frame plus a side panel with objects, status, and chat."""

    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _HEADING_SCALE = 0.6
    _TEXT_SCALE = 0.45
    _LINE_HEIGHT = 18
    _MARGIN = 16
    _BLINK_HZ = 2.0

    _STATE_LABELS: Dict[VoiceState, Tuple[str, Tuple[int, int, int]]] = {
        VoiceState.RECORDING: ("Recording", (0, 0, 255)),
        VoiceState.TRANSCRIBING: ("Transcribing...", (0, 165, 255)),
        VoiceState.THINKING: ("Thinking...", (0, 210, 255)),
        VoiceState.SPEAKING: ("Speaking...", (0, 200, 0)),
    }

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

    def show(
        self,
        annotated_frame: np.ndarray,
        descriptions: List[ObjectDescription],
        voice_state: VoiceState = VoiceState.IDLE,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> None:
        panel = self._render_panel(annotated_frame.shape[0], descriptions, voice_state, chat_history or [])
        canvas = np.hstack([annotated_frame, panel])
        cv2.imshow(self._window_name, canvas)

    def poll_key(self) -> Optional[str]:
        """Return the character key pressed since the last poll, if any.

        Also pumps the window's event loop, so this doubles as the call
        that makes a preceding show() actually paint to screen.
        """
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

    def _render_panel(
        self,
        height: int,
        descriptions: List[ObjectDescription],
        voice_state: VoiceState,
        chat_history: List[ChatMessage],
    ) -> np.ndarray:
        panel = np.full((height, self._panel_width, 3), self._panel_bg_color, dtype=np.uint8)
        text_width = self._panel_width - 2 * self._MARGIN
        bottom = height - self._MARGIN

        y = self._MARGIN + self._LINE_HEIGHT
        y = self._render_status_banner(panel, y, voice_state) + int(self._LINE_HEIGHT * 0.5)

        cv2.putText(
            panel, "Detected Objects", (self._MARGIN, y),
            self._FONT, self._HEADING_SCALE, self._heading_color, 2,
        )
        y += int(self._LINE_HEIGHT * 1.5)

        objects_max_y = y + int((bottom - y) * 0.5)
        y = self._render_objects(panel, y, objects_max_y, text_width, descriptions)

        y = max(y, objects_max_y) + int(self._LINE_HEIGHT * 0.5)
        cv2.putText(
            panel, "Chat History", (self._MARGIN, y),
            self._FONT, self._HEADING_SCALE, self._heading_color, 2,
        )
        y += int(self._LINE_HEIGHT * 1.5)

        self._render_chat(panel, y, bottom, text_width, chat_history)

        return panel

    def _render_status_banner(self, panel: np.ndarray, y: int, voice_state: VoiceState) -> int:
        if voice_state == VoiceState.IDLE:
            cv2.putText(
                panel, "Ready -- press 'v' to talk", (self._MARGIN, y),
                self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
            )
            return y

        label, color = self._STATE_LABELS[voice_state]
        dot_radius = 6
        dot_center = (self._MARGIN + dot_radius, y - dot_radius)
        # Recording blinks (many real frames get drawn while listening);
        # other states are steady since each is shown for one blocking step.
        if voice_state != VoiceState.RECORDING or self._blink_on():
            cv2.circle(panel, dot_center, dot_radius, color, -1)
        cv2.putText(
            panel, label, (self._MARGIN + 2 * dot_radius + 8, y),
            self._FONT, self._TEXT_SCALE, color, 1,
        )
        return y

    def _blink_on(self) -> bool:
        return int(time.time() * self._BLINK_HZ) % 2 == 0

    def _render_objects(
        self,
        panel: np.ndarray,
        y: int,
        max_y: int,
        text_width: int,
        descriptions: List[ObjectDescription],
    ) -> int:
        if not descriptions:
            cv2.putText(
                panel, "No objects detected", (self._MARGIN, y),
                self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
            )
            return y + self._LINE_HEIGHT

        for index, description in enumerate(descriptions, start=1):
            heading = f"{index}. {description.label.capitalize()} ({description.confidence:.0%})"
            body = (
                f"{description.size_label}, {description.width_px}x{description.height_px}px, "
                f"predominantly {description.color_name}. {description.feature_text}"
            )
            lines = [heading] + self._wrap_text(body, self._TEXT_SCALE, text_width)

            if y + len(lines) * self._LINE_HEIGHT > max_y:
                remaining = len(descriptions) - index + 1
                cv2.putText(
                    panel, f"+ {remaining} more", (self._MARGIN, y),
                    self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
                )
                return y + self._LINE_HEIGHT

            cv2.putText(panel, lines[0], (self._MARGIN, y), self._FONT, self._TEXT_SCALE, self._text_color, 1)
            y += self._LINE_HEIGHT
            for line in lines[1:]:
                cv2.putText(panel, line, (self._MARGIN, y), self._FONT, self._TEXT_SCALE, self._subtext_color, 1)
                y += self._LINE_HEIGHT
            y += self._LINE_HEIGHT // 2

        return y

    def _render_chat(
        self,
        panel: np.ndarray,
        top: int,
        bottom: int,
        text_width: int,
        chat_history: List[ChatMessage],
    ) -> None:
        if not chat_history:
            cv2.putText(
                panel, "No conversation yet", (self._MARGIN, top),
                self._FONT, self._TEXT_SCALE, self._subtext_color, 1,
            )
            return

        available = bottom - top
        used = 0
        blocks: List[Tuple[Tuple[int, int, int], List[str]]] = []
        for message in reversed(chat_history):
            label = "You" if message.speaker == "you" else "Assistant"
            color = self._text_color if message.speaker == "you" else self._heading_color
            lines = [f"{label}:"] + self._wrap_text(message.text, self._TEXT_SCALE, text_width)
            block_height = len(lines) * self._LINE_HEIGHT + self._LINE_HEIGHT // 2

            if used + block_height > available:
                break
            blocks.append((color, lines))
            used += block_height

        y = top
        for color, lines in reversed(blocks):
            cv2.putText(panel, lines[0], (self._MARGIN, y), self._FONT, self._TEXT_SCALE, color, 1)
            y += self._LINE_HEIGHT
            for line in lines[1:]:
                cv2.putText(panel, line, (self._MARGIN, y), self._FONT, self._TEXT_SCALE, self._subtext_color, 1)
                y += self._LINE_HEIGHT
            y += self._LINE_HEIGHT // 2

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
