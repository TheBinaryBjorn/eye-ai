"""Speaker hardware interaction module.

Sole responsibility: play an audio clip through the system speakers. Knows
nothing about how the audio was synthesized.
"""

from __future__ import annotations

from typing import Optional

import sounddevice as sd

from src.microphone.microphone_stream import AudioClip


class SpeakerOutput:
    """Owns playback to the default (or a chosen) output device."""

    def __init__(self, device: Optional[int] = None) -> None:
        self._device = device

    def play(self, clip: AudioClip) -> None:
        sd.play(clip.samples, samplerate=clip.sample_rate, device=self._device)
        sd.wait()
