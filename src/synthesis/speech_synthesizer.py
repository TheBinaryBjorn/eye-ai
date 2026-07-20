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
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return AudioClip(samples=np.zeros(0, dtype=np.float32), sample_rate=22050)

        samples = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        return AudioClip(samples=samples, sample_rate=chunks[0].sample_rate)
