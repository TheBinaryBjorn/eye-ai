"""Entry point: wires the vision pipeline and the voice pipeline together.

Vision:  CameraStream -> ObjectDetector -> ObjectDescriber -> StreamDisplay
Voice:   MicrophoneStream -> SpeechRecognizer -> ConversationalAgent
         -> SpeechSynthesizer -> SpeakerOutput

Every module above only exchanges plain data with its neighbors (frames,
Detection/ObjectDescription records, AudioClip, text) -- this file is the
only place that knows how they fit together.
"""

from __future__ import annotations

from typing import List

from src import config
from src.camera.camera_stream import CameraStream
from src.description.object_describer import ObjectDescriber, ObjectDescription
from src.detection.object_detector import ObjectDetector
from src.display.stream_display import StreamDisplay
from src.llm.conversational_agent import ConversationalAgent
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

    visible_objects: List[ObjectDescription] = []

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
            display.show(annotated_frame, visible_objects)

            key = display.poll_key()
            if key == config.QUIT_KEY:
                break
            if key == config.VOICE_KEY:
                _handle_voice_key(microphone, recognizer, agent, synthesizer, speaker, visible_objects)


def _handle_voice_key(
    microphone: MicrophoneStream,
    recognizer: SpeechRecognizer,
    agent: ConversationalAgent,
    synthesizer: SpeechSynthesizer,
    speaker: SpeakerOutput,
    visible_objects: List[ObjectDescription],
) -> None:
    """Push-to-talk: first press starts recording, second press sends it."""
    if not microphone.is_recording():
        print("[voice] listening... press 'v' again to stop")
        microphone.start_recording()
        return

    print("[voice] transcribing...")
    clip = microphone.stop_recording()
    user_message = recognizer.transcribe(clip)
    if not user_message:
        print("[voice] didn't catch that, try again")
        return

    print(f"[voice] you: {user_message}")
    print("[voice] thinking...")
    reply = agent.ask(user_message, visible_objects)
    print(f"[voice] assistant: {reply}")

    speaker.play(synthesizer.synthesize(reply))


if __name__ == "__main__":
    main()
