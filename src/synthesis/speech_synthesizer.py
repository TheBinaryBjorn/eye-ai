"""Speech synthesis module.

Sole responsibility: turn text into an audio clip. Knows nothing about the
LLM that produced the text or how the clip will be played back (the
speaker module).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from piper import PiperVoice

from src.microphone.microphone_stream import AudioClip


class SpeechSynthesizer:
    """Wraps a local Piper TTS voice to synthesize speech audio."""

    def __init__(self, model_path: str, config_path: Optional[str] = None) -> None:
        self._voice = PiperVoice.load(model_path, config_path=config_path)

    def synthesize(self, text: str) -> AudioClip:
        raw_audio = b"".join(self._voice.synthesize_stream_raw(text))
        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        return AudioClip(samples=samples, sample_rate=self._voice.config.sample_rate)
