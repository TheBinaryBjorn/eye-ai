"""Speech transcription module.

Sole responsibility: turn a recorded audio clip into text. Knows nothing
about how the audio was captured (microphone) or how the text will be used
(the LLM).
"""

from __future__ import annotations

from faster_whisper import WhisperModel

from src.microphone.microphone_stream import AudioClip


class SpeechRecognizer:
    """Wraps a local faster-whisper model to transcribe audio clips."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, clip: AudioClip) -> str:
        if clip.samples.size == 0:
            return ""

        segments, _info = self._model.transcribe(clip.samples, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()
