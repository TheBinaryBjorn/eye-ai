"""Entry point: wires the vision pipeline and the voice pipeline together.

Vision:  CameraStream -> ObjectDetector -> ObjectDescriber -> StreamDisplay
Voice:   MicrophoneStream -> SpeechRecognizer -> ConversationalAgent
         -> SpeechSynthesizer -> SpeakerOutput

Every module above only exchanges plain data with its neighbors (frames,
Detection/ObjectDescription records, AudioClip, ChatMessage, text) -- this
file is the only place that knows how they fit together.

Detection + description run on their own background thread (see
_PerceptionWorker), so the display loop only has to read a frame, draw the
most recently available boxes on it, and show it -- it runs at camera FPS
and stays smooth no matter how slow YOLO is or what the voice pipeline is
doing on other threads.

The transcribe/think/speak steps of a voice turn also run on a background
thread (see _VoiceSession) so the video loop -- and the live status
banner -- never stops while a turn is in progress.

Voice input is true hold-to-talk: recording starts the instant VOICE_KEY
is physically pressed down and stops the instant it's released. cv2's own
key polling (used for QUIT_KEY) only ever reports key-*down* events, never
release, so it can't do this -- VOICE_KEY's actual physical state is
polled directly via the Windows API instead (see _is_voice_key_down).
This is Windows-only; there's no cross-platform requirement here.
"""

from __future__ import annotations

import ctypes
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

_KEY_DOWN_FLAG = 0x8000


def _is_voice_key_down() -> bool:
    vk_code = ord(config.VOICE_KEY.upper())
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & _KEY_DOWN_FLAG)


class _PerceptionWorker:
    """Runs detection + description on a background thread.

    The display loop hands it the newest frame via submit() and reads back
    the most recent results via latest(), never blocking on YOLO itself.
    Only the latest submitted frame is ever processed -- if detection can't
    keep up with the camera, intermediate frames are simply skipped, which
    is what we want (we always want the freshest possible detections, not a
    backlog of stale ones).
    """

    def __init__(self, detector: ObjectDetector, describer: ObjectDescriber) -> None:
        self._detector = detector
        self._describer = describer
        self._lock = threading.Lock()
        self._pending_frame: "np.ndarray | None" = None
        self._new_frame = threading.Event()
        self._detections: List = []
        self._descriptions: List[ObjectDescription] = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame
        self._new_frame.set()

    def latest(self) -> "tuple[List, List[ObjectDescription]]":
        with self._lock:
            return self._detections, list(self._descriptions)

    def stop(self) -> None:
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while self._running:
            self._new_frame.wait()
            self._new_frame.clear()
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue

            detections = self._detector.detect(frame)
            descriptions = self._describer.describe(frame, detections)
            with self._lock:
                self._detections = detections
                self._descriptions = descriptions


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
    perception = _PerceptionWorker(detector, describer)
    perception.start()

    try:
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

                # Detection runs on the worker thread; the display loop just
                # draws the most recent boxes on the live frame and shows it.
                perception.submit(frame)
                detections, visible_objects = perception.latest()
                annotated_frame = detector.annotate(frame, detections)
                display.show(annotated_frame, visible_objects, session.state, agent.history)

                key = display.poll_key()
                if key == config.QUIT_KEY:
                    break

                _handle_voice_key_state(
                    microphone, recognizer, agent, synthesizer, speaker, session, visible_objects,
                )
    finally:
        perception.stop()


def _handle_voice_key_state(
    microphone: MicrophoneStream,
    recognizer: SpeechRecognizer,
    agent: ConversationalAgent,
    synthesizer: SpeechSynthesizer,
    speaker: SpeakerOutput,
    session: _VoiceSession,
    visible_objects: List[ObjectDescription],
) -> None:
    """True hold-to-talk: start on the press edge, stop on the release edge.

    Only reacts to the two edges that actually matter -- IDLE-and-now-held
    starts a recording, RECORDING-and-now-released stops it -- so holding
    the key doesn't repeatedly retrigger anything, and the key is ignored
    entirely while a previous turn is still being processed in the
    background (state is TRANSCRIBING/THINKING/SPEAKING).
    """
    key_down = _is_voice_key_down()
    state = session.state

    if key_down and state == VoiceState.IDLE:
        microphone.start_recording()
        session.set_state(VoiceState.RECORDING)
        return

    if not key_down and state == VoiceState.RECORDING:
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
