"""Entry point: wires the vision pipeline and the voice pipeline together.

Vision:  CameraStream -> ObjectDetector -> ObjectDescriber -> StreamDisplay
Voice:   MicrophoneStream -> SpeechRecognizer -> ConversationalAgent
         -> SpeechSynthesizer -> SpeakerOutput

Every module above only exchanges plain data with its neighbors (frames,
Detection/ObjectDescription records, AudioClip, ChatMessage, text) -- this
file is the only place that knows how they fit together.

The transcribe/think/speak steps of a voice turn run on a background
thread (see _VoiceSession) so the video loop -- and the live status
banner -- never stops while a turn is in progress.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np

from src import config
from src.camera.camera_stream import CameraStream
from src.description.object_describer import ObjectDescriber, ObjectDescription
from src.detection.object_detector import ObjectDetector
from src.display.stream_display import StreamDisplay, VoiceState
from src.llm.conversational_agent import ConversationalAgent
from src.microphone.microphone_stream import AudioClip, MicrophoneStream
from src.speaker.speaker_output import SpeakerOutput
from src.synthesis.speech_synthesizer import SpeechSynthesizer
from src.transcription.speech_recognizer import SpeechRecognizer

# Below this peak sample amplitude (of a [-1.0, 1.0] clip), treat a
# recording as effectively silent and warn instead of sending it to
# Whisper, which tends to hallucinate "you"/"thank you" on silence.
_SILENT_PEAK_THRESHOLD = 0.02


class _VoiceSession:
    """Tracks which phase of a voice turn is in progress, safely across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = VoiceState.IDLE

    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    def set_state(self, state: VoiceState) -> None:
        with self._lock:
            self._state = state


def main() -> None:
    detector = ObjectDetector(config.MODEL_NAME, config.CONFIDENCE_THRESHOLD, config.BOX_COLOR)
    describer = ObjectDescriber()

    microphone = MicrophoneStream(config.MIC_SAMPLE_RATE, device=config.MIC_DEVICE)
    recognizer = SpeechRecognizer(config.WHISPER_MODEL_SIZE)
    agent = ConversationalAgent(config.LLM_MODEL, config.OLLAMA_HOST)
    synthesizer = SpeechSynthesizer(config.PIPER_MODEL_PATH, config.PIPER_CONFIG_PATH)
    speaker = SpeakerOutput()

    session = _VoiceSession()

    with CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT) as camera, \
            StreamDisplay(
                config.WINDOW_NAME,
                config.PANEL_WIDTH,
                config.PANEL_BG_COLOR,
                config.PANEL_HEADING_COLOR,
                config.PANEL_TEXT_COLOR,
                config.PANEL_SUBTEXT_COLOR,
            ) as display:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            annotated_frame, detections = detector.detect(frame)
            visible_objects = describer.describe(frame, detections)
            display.show(annotated_frame, visible_objects, session.state, agent.history)

            key = display.poll_key()
            if key == config.QUIT_KEY:
                break
            if key == config.VOICE_KEY:
                _handle_voice_key(microphone, recognizer, agent, synthesizer, speaker, session, visible_objects)


def _handle_voice_key(
    microphone: MicrophoneStream,
    recognizer: SpeechRecognizer,
    agent: ConversationalAgent,
    synthesizer: SpeechSynthesizer,
    speaker: SpeakerOutput,
    session: _VoiceSession,
    visible_objects: List[ObjectDescription],
) -> None:
    """Push-to-talk: first press starts recording, second press sends it.

    The send side (transcribe -> think -> speak) runs on a background
    thread so the caller (the video loop) is never blocked.
    """
    state = session.state
    if state == VoiceState.IDLE:
        microphone.start_recording()
        session.set_state(VoiceState.RECORDING)
        return

    if state != VoiceState.RECORDING:
        return  # a previous turn is still being processed; ignore the key

    clip = microphone.stop_recording()
    session.set_state(VoiceState.TRANSCRIBING)
    threading.Thread(
        target=_run_voice_turn,
        args=(clip, recognizer, agent, synthesizer, speaker, session, visible_objects),
        daemon=True,
    ).start()


def _run_voice_turn(
    clip: AudioClip,
    recognizer: SpeechRecognizer,
    agent: ConversationalAgent,
    synthesizer: SpeechSynthesizer,
    speaker: SpeakerOutput,
    session: _VoiceSession,
    visible_objects: List[ObjectDescription],
) -> None:
    try:
        duration = clip.samples.size / clip.sample_rate if clip.sample_rate else 0.0
        peak = float(np.abs(clip.samples).max()) if clip.samples.size else 0.0
        print(f"[voice] recorded {duration:.1f}s, peak level {peak:.3f}")
        if peak < _SILENT_PEAK_THRESHOLD:
            print(
                "[voice] recording seems silent -- check that the right microphone is "
                "selected (see MIC_DEVICE in src/config.py) and that it isn't muted"
            )

        user_message = recognizer.transcribe(clip)
        if not user_message:
            print("[voice] didn't catch that, try again")
            return
        print(f"[voice] you: {user_message}")

        session.set_state(VoiceState.THINKING)
        reply = agent.ask(user_message, visible_objects)
        print(f"[voice] assistant: {reply}")

        session.set_state(VoiceState.SPEAKING)
        speaker.play(synthesizer.synthesize(reply))
    finally:
        session.set_state(VoiceState.IDLE)


if __name__ == "__main__":
    main()
