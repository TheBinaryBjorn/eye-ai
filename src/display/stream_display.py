"""Output stream module.

Sole responsibility: composite the annotated frame, the object descriptions,
the voice pipeline's current phase, and the chat history into a single
window, and report key presses from that window (quit, voice trigger,
etc). Knows nothing about how frames are captured, detected, described, or
how the voice pipeline actually works -- it only renders the plain data
(and state label) it's handed.

The side panel is styled as frosted "liquid glass": its backdrop is a
blurred, brightened copy of the live frame (so it reads as glass frosting
the scene, the way a macOS sidebar frosts the wallpaper behind it), with
translucent glass cards and anti-aliased text drawn on top via Pillow.
"""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.description.object_describer import ObjectDescription
from src.llm.conversational_agent import ChatMessage

_FONTS_DIR = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
_FONT_REGULAR = os.path.join(_FONTS_DIR, "segoeui.ttf")
_FONT_SEMIBOLD = os.path.join(_FONTS_DIR, "seguisb.ttf")
_FONT_BOLD = os.path.join(_FONTS_DIR, "segoeuib.ttf")

# RGBA colors (Pillow works in RGB; the panel is converted back to BGR at the end).
_WHITE = (255, 255, 255)
_TEXT_PRIMARY = (255, 255, 255, 242)
_TEXT_SECONDARY = (255, 255, 255, 200)
_TEXT_FAINT = (255, 255, 255, 160)
_SHADOW = (0, 0, 0, 115)
_CARD_FILL = (255, 255, 255, 30)
_CARD_FILL_STRONG = (255, 255, 255, 46)
_CARD_BORDER = (255, 255, 255, 70)
_ASSISTANT_ACCENT = (120, 215, 255, 235)


class VoiceState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


# label + accent RGB per state
_STATE_STYLE = {
    VoiceState.IDLE: ("Ready · hold V to talk", (200, 210, 220)),
    VoiceState.RECORDING: ("Recording", (255, 69, 58)),
    VoiceState.TRANSCRIBING: ("Transcribing", (255, 159, 10)),
    VoiceState.THINKING: ("Thinking", (255, 214, 10)),
    VoiceState.SPEAKING: ("Speaking", (48, 209, 88)),
}


class StreamDisplay:
    """Shows the annotated frame plus a frosted-glass side panel."""

    _MARGIN = 24
    _BLINK_HZ = 2.0

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
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    # ----- public API -------------------------------------------------------

    def show(
        self,
        annotated_frame: np.ndarray,
        descriptions: List[ObjectDescription],
        voice_state: VoiceState = VoiceState.IDLE,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> None:
        panel = self._render_panel(annotated_frame, descriptions, voice_state, chat_history or [])
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

    # ----- rendering --------------------------------------------------------

    def _render_panel(
        self,
        frame: np.ndarray,
        descriptions: List[ObjectDescription],
        voice_state: VoiceState,
        chat_history: List[ChatMessage],
    ) -> np.ndarray:
        height = frame.shape[0]
        width = self._panel_width

        base = self._frost_backdrop(frame, width, height)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # bright glass edge where the panel meets the video
        draw.line((1, 0, 1, height), fill=(255, 255, 255, 75), width=1)

        mx = self._MARGIN
        content_w = width - 2 * mx
        y = mx + 4

        # Header
        self._text(draw, (mx, y), "eye·ai", self._font(_FONT_BOLD, 30), _TEXT_PRIMARY)
        y += 40
        self._text(draw, (mx, y), "Live vision + voice", self._font(_FONT_REGULAR, 14), _TEXT_SECONDARY)
        y += 30

        # Status pill
        y = self._draw_status(draw, mx, y, content_w, voice_state) + 22

        # Split the rest between objects and chat.
        usable_bottom = height - mx
        remaining = usable_bottom - (y + 20)
        objects_budget = int(remaining * 0.5)

        self._section_label(draw, mx, y, "DETECTED OBJECTS")
        y += 22
        y = self._draw_objects(draw, mx, y, y + objects_budget, content_w, descriptions)

        y += 8
        self._section_label(draw, mx, y, "CONVERSATION")
        y += 22
        self._draw_chat(draw, mx, y, usable_bottom, content_w, chat_history)

        out = Image.alpha_composite(base, overlay).convert("RGB")
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    def _frost_backdrop(self, frame: np.ndarray, width: int, height: int) -> Image.Image:
        """Blurred, brightened crop of the live frame -> the 'glass' behind the UI."""
        h, w = frame.shape[:2]
        crop = frame[:, w - width:] if w >= width else cv2.resize(frame, (width, height))

        # Downscale -> heavy blur -> upscale is fast and gives a smooth frost.
        small = cv2.resize(crop, (max(width // 4, 1), max(height // 4, 1)))
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=9)
        blurred = cv2.resize(small, (width, height)).astype(np.float32)

        # A light white veil (glass frost) plus a cool dark scrim so white
        # text keeps contrast no matter how bright the scene behind is.
        frost = blurred * 0.74 + 255.0 * 0.12
        navy = np.array([44, 34, 56], np.float32)  # BGR, subtle purple-navy
        frost = frost * 0.80 + navy * 0.20
        frost = np.clip(frost, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(frost, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGBA")

    def _draw_status(self, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, state: VoiceState) -> int:
        label, accent = _STATE_STYLE[state]
        height = 40
        self._glass_card(draw, (x, y, x + w, y + height), radius=height // 2, strong=(state != VoiceState.IDLE))

        cy = y + height // 2
        dot_x = x + 20
        show_dot = state != VoiceState.RECORDING or self._blink_on()
        if show_dot:
            glow = (*accent, 70)
            draw.ellipse((dot_x - 9, cy - 9, dot_x + 9, cy + 9), fill=glow)
            draw.ellipse((dot_x - 5, cy - 5, dot_x + 5, cy + 5), fill=(*accent, 255))

        font = self._font(_FONT_SEMIBOLD, 15)
        _, top, _, bottom = font.getbbox(label)
        self._text(draw, (dot_x + 18, cy - (bottom + top) // 2 - top), label, font, _TEXT_PRIMARY)
        return y + height

    def _draw_objects(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        max_y: int,
        w: int,
        descriptions: List[ObjectDescription],
    ) -> int:
        if not descriptions:
            self._text(draw, (x, y), "Nothing detected yet", self._font(_FONT_REGULAR, 13), _TEXT_FAINT)
            return y + 22

        title_font = self._font(_FONT_SEMIBOLD, 16)
        meta_font = self._font(_FONT_REGULAR, 12)
        conf_font = self._font(_FONT_SEMIBOLD, 12)
        pad = 12
        card_h = pad * 2 + 20 + 4 + 15

        for index, d in enumerate(descriptions, start=1):
            if y + card_h > max_y:
                left = len(descriptions) - index + 1
                self._text(draw, (x, y + 2), f"+ {left} more", self._font(_FONT_REGULAR, 12), _TEXT_FAINT)
                return y + 20
            self._glass_card(draw, (x, y, x + w, y + card_h), radius=14)

            self._text(draw, (x + pad, y + pad), d.label.capitalize(), title_font, _TEXT_PRIMARY)
            conf = f"{d.confidence:.0%}"
            cw = draw.textlength(conf, font=conf_font)
            self._text(draw, (x + w - pad - cw, y + pad + 3), conf, conf_font, _ASSISTANT_ACCENT)

            meta = f"{d.size_label} · {d.width_px}×{d.height_px}px · {d.color_name}"
            self._text(draw, (x + pad, y + pad + 24), meta, meta_font, _TEXT_SECONDARY)
            y += card_h + 10

        return y

    def _draw_chat(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        top: int,
        bottom: int,
        w: int,
        chat_history: List[ChatMessage],
    ) -> None:
        if not chat_history:
            self._text(draw, (x, top), "No conversation yet", self._font(_FONT_REGULAR, 13), _TEXT_FAINT)
            return

        speaker_font = self._font(_FONT_SEMIBOLD, 12)
        text_font = self._font(_FONT_REGULAR, 13)
        pad = 11
        line_h = 18

        # Build bubbles newest-first until we run out of vertical room, then
        # draw them top-down so the newest sits at the bottom.
        blocks: List[Tuple[bool, List[str], int]] = []
        used = 0
        available = bottom - top
        for msg in reversed(chat_history):
            is_you = msg.speaker == "you"
            lines = self._wrap(draw, msg.text, text_font, w - 2 * pad)
            block_h = pad * 2 + 16 + len(lines) * line_h + 10
            if used + block_h > available:
                break
            blocks.append((is_you, lines, block_h))
            used += block_h

        y = bottom - used
        for is_you, lines, block_h in reversed(blocks):
            self._glass_card(draw, (x, y, x + w, y + block_h - 10), radius=14, strong=is_you)
            label = "You" if is_you else "Assistant"
            self._text(draw, (x + pad, y + pad), label, speaker_font,
                       _TEXT_SECONDARY if is_you else _ASSISTANT_ACCENT)
            ty = y + pad + 18
            for line in lines:
                self._text(draw, (x + pad, ty), line, text_font, _TEXT_PRIMARY)
                ty += line_h
            y += block_h

    # ----- drawing primitives ----------------------------------------------

    def _glass_card(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], radius: int, strong: bool = False) -> None:
        fill = _CARD_FILL_STRONG if strong else _CARD_FILL
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=_CARD_BORDER, width=1)
        # thin bright top edge for a subtle glass highlight
        x0, y0, x1, _ = xy
        draw.line((x0 + radius, y0 + 1, x1 - radius, y0 + 1), fill=(255, 255, 255, 90), width=1)

    def _text(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str,
              font: ImageFont.FreeTypeFont, fill: Tuple[int, ...]) -> None:
        x, y = xy
        draw.text((x, y + 1), text, font=font, fill=_SHADOW)
        draw.text((x, y), text, font=font, fill=fill)

    def _section_label(self, draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
        font = self._font(_FONT_BOLD, 11)
        cx = x
        for ch in text:
            draw.text((cx, y + 1), ch, font=font, fill=_SHADOW)
            draw.text((cx, y), ch, font=font, fill=_TEXT_SECONDARY)
            cx += draw.textlength(ch, font=font) + 2

    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= max_w or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(path, size)
            except OSError:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def _blink_on(self) -> bool:
        return int(time.time() * self._BLINK_HZ) % 2 == 0
