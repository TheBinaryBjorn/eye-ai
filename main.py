"""Entry point: wires the vision pipeline and the voice pipeline together.

Vision:  CameraStream -> ObjectDetector -> ObjectDescriber -> StreamDisplay
Voice:   MicrophoneStream -> SpeechRecognizer -> ConversationalAgent
         -> SpeechSynthesizer -> SpeakerOutput

Every module above only exchanges plain data with its neighbors (frames,
Detection/ObjectDescription records, AudioClip, ChatMessage, text) -- this
file is the only place that knows how they fit together, including which
VoiceState to show the display while a voice turn is in progress.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src import config
from src.camera.camera_stream import CameraStream
from src.description.object_describer import ObjectDescriber, ObjectDescription
from src.detection.object_detector import ObjectDetector
from src.display.stream_display import StreamDisplay, VoiceState
from src.llm.conversational_agent import ChatMessage, ConversationalAgent
from src.microphone.microphone_stream import MicrophoneStream
from src.speaker.speaker_output import SpeakerOutput
from src.synthesis.speech_synthesizer import SpeechSynthesizer
from src.transcription.speech_recognizer import SpeechRecognizer


def main() -> None:
    detector = ObjectDetector(config.MODEL_NAME, config.CONFIDENCE_THRESHOLD, config.BOX_COLOR)
    describer = ObjectDescriber()

    microphone = MicrophoneStream(config.MIC_SAMPLE_RATE)
    recognizer = SpeechRecognizer(config.WHISPER_MODEL_SIZE)
    agent = ConversationalAgent(config.LLM_MODEL, config.OLLAMA_HOST)
    synthesizer = SpeechSynthesizer(config.PIPER_MODEL_PATH, config.PIPER_CONFIG_PATH)
    speaker = SpeakerOutput()

    voice_state = VoiceState.IDLE

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
            display.show(annotated_frame, visible_objects, voice_state, agent.history)

            key = display.poll_key()
            if key == config.QUIT_KEY:
                break
            if key == config.VOICE_KEY:
                voice_state = _handle_voice_key(
                    microphone, recognizer, agent, synthesizer, speaker,
                    display, annotated_frame, visible_objects,
                )


def _handle_voice_key(
    microphone: MicrophoneStream,
    recognizer: SpeechRecognizer,
    agent: ConversationalAgent,
    synthesizer: SpeechSynthesizer,
    speaker: SpeakerOutput,
    display: StreamDisplay,
    annotated_frame: np.ndarray,
    visible_objects: List[ObjectDescription],
) -> VoiceState:
    """Push-to-talk: first press starts recording, second press sends it.

    Returns the VoiceState the main loop should keep showing on the next
    frame. Recording is genuinely live (the main loop keeps redrawing
    while the mic buffers in the background); the transcribe/think/speak
    steps are blocking, so we redraw once before each one so its status
    is at least visible on screen for the whole step.
    """
    if not microphone.is_recording():
        microphone.start_recording()
        return VoiceState.RECORDING

    clip = microphone.stop_recording()

    _redraw(display, annotated_frame, visible_objects, VoiceState.TRANSCRIBING, agent.history)
    user_message = recognizer.transcribe(clip)
    if not user_message:
        print("[voice] didn't catch that, try again")
        return VoiceState.IDLE
    print(f"[voice] you: {user_message}")

    _redraw(display, annotated_frame, visible_objects, VoiceState.THINKING, agent.history)
    reply = agent.ask(user_message, visible_objects)
    print(f"[voice] assistant: {reply}")

    _redraw(display, annotated_frame, visible_objects, VoiceState.SPEAKING, agent.history)
    speaker.play(synthesizer.synthesize(reply))

    return VoiceState.IDLE


def _redraw(
    display: StreamDisplay,
    annotated_frame: np.ndarray,
    visible_objects: List[ObjectDescription],
    voice_state: VoiceState,
    chat_history: List[ChatMessage],
) -> None:
    """Push a frame to screen immediately, so a status change is visible
    before a blocking pipeline step runs."""
    display.show(annotated_frame, visible_objects, voice_state, chat_history)
    display.poll_key()


if __name__ == "__main__":
    main()
