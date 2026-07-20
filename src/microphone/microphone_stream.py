"""Microphone hardware interaction module.

Sole responsibility: capture raw audio from the microphone and hand back
samples. Knows nothing about transcription, the LLM, or how the clip will
eventually be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # float32 mono samples in [-1.0, 1.0]
    sample_rate: int


class MicrophoneStream:
    """Owns the microphone device and records push-to-talk style clips."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = self._resolve_device(device)
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None

    def start_recording(self) -> None:
        """Begin buffering audio. Call stop_recording() to get the clip back."""
        if self._stream is not None:
            raise RuntimeError("Recording already in progress")

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
            dtype="float32",
            callback=self._on_audio_block,
        )
        self._stream.start()

    def stop_recording(self) -> AudioClip:
        """Stop buffering and return everything captured since start_recording()."""
        if self._stream is None:
            raise RuntimeError("No recording in progress")

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if self._frames:
            samples = np.concatenate(self._frames, axis=0).reshape(-1)
        else:
            samples = np.zeros(0, dtype=np.float32)
        return AudioClip(samples=samples, sample_rate=self._sample_rate)

    def is_recording(self) -> bool:
        return self._stream is not None

    def _on_audio_block(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        self._frames.append(indata.copy())

    @staticmethod
    def _resolve_device(device: Optional[Union[int, str]]) -> Optional[int]:
        """Resolve a device index, a name substring, or None (system default).

        A numeric index is fragile -- it can shift after a driver update, a
        device is unplugged, or the same config is used on another machine.
        A name substring is stable across all of those, at the cost of one
        extra lookup at startup.
        """
        if device is None or isinstance(device, int):
            return device

        for index, info in enumerate(sd.query_devices()):
            if device.lower() in info["name"].lower() and info["max_input_channels"] > 0:
                return index

        raise ValueError(
            f"No input device matching {device!r} found. Run "
            '`python -c "import sounddevice as sd; print(sd.query_devices())"` to list devices.'
        )
