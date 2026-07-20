# eye-ai

Reads a webcam stream, recognizes objects in each frame, draws red bounding
boxes around them, and shows a single window with the annotated video on
the left and a description panel (color, size, an interesting fact per
object) on the right.

It also doubles as a voice assistant: press `v` to ask a local LLM a
question by voice, and it answers out loud, grounded in what it currently
sees through the camera.

## Architecture

Each module owns exactly one concern. Two independent pipelines meet only
in `main.py`, which is the single place that wires them together.

**Vision pipeline** (camera → detect → describe → display):

- [src/camera/camera_stream.py](src/camera/camera_stream.py) — `CameraStream`: opens the webcam and returns raw frames. Knows nothing about detection, description, or display.
- [src/detection/object_detector.py](src/detection/object_detector.py) — `ObjectDetector`: takes a frame, runs YOLOv8 detection, draws red boxes/labels, returns the annotated frame and the raw `Detection` list (label, confidence, box). Knows nothing about the camera, descriptions, or display.
- [src/description/object_describer.py](src/description/object_describer.py) — `ObjectDescriber`: takes the original frame and `Detection` list, and for each one derives a dominant color, a size classification, and a short fact, returning `ObjectDescription` records. Knows nothing about the camera, detection internals, or display.
- [src/display/stream_display.py](src/display/stream_display.py) — `StreamDisplay`: composites the annotated frame and the `ObjectDescription` list into one window (video + side panel), and reports raw key presses from that window. Knows nothing about capture, detection, description, or what any given key means.

**Voice pipeline** (microphone → transcribe → llm → synthesize → speaker), triggered by pressing `v` in the video window:

- [src/microphone/microphone_stream.py](src/microphone/microphone_stream.py) — `MicrophoneStream`: hardware interaction only. Records raw audio while active and returns an `AudioClip` (samples + sample rate). Knows nothing about transcription.
- [src/transcription/speech_recognizer.py](src/transcription/speech_recognizer.py) — `SpeechRecognizer`: wraps a local `faster-whisper` model, turns an `AudioClip` into text. Knows nothing about the microphone or the LLM.
- [src/llm/conversational_agent.py](src/llm/conversational_agent.py) — `ConversationalAgent`: wraps a local Ollama model, holds the chat history, and answers a question given the current `ObjectDescription` list as scene context. Exposes a clean `history` property (`ChatMessage` records, no system prompt or scene-context prefix) purely for display. Knows nothing about audio hardware, transcription, synthesis, or display.
- [src/synthesis/speech_synthesizer.py](src/synthesis/speech_synthesizer.py) — `SpeechSynthesizer`: wraps a local Piper TTS voice, turns the LLM's text reply into an `AudioClip`. Knows nothing about the LLM or playback.
- [src/speaker/speaker_output.py](src/speaker/speaker_output.py) — `SpeakerOutput`: hardware interaction only. Plays an `AudioClip` through the system speakers. Knows nothing about how it was synthesized.

- [main.py](main.py) — runs the vision loop every frame, and on a `v` key press runs one push-to-talk turn through the voice pipeline using the vision pipeline's latest `ObjectDescription` list as context. Tracks which `VoiceState` (recording/transcribing/thinking/speaking) the display should show, and passes the agent's `history` to it each frame so the side panel stays current.

The side panel (still rendered by `StreamDisplay`, still fed nothing but
plain data) has three parts: a status banner reflecting the current
`VoiceState`, the detected-objects list, and the tail of the chat history
that fits in the remaining space. `StreamDisplay` decides how to draw each
`VoiceState`; it doesn't know what a state means for the pipeline, just
what color/label to show for it.

Because every module only exchanges plain data (numpy frames, `Detection` /
`ObjectDescription` / `AudioClip` dataclasses, or text) with its neighbors,
any one of them can be swapped independently — a different camera, a
different detection model, a different STT/TTS engine, or a different LLM
backend — without touching the others.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

The first run of the vision pipeline downloads the `yolov8n.pt` pretrained
weights automatically. The voice pipeline needs three more things set up
once:

1. **Ollama**, running locally with a model pulled:
   ```bash
   ollama pull llama3.2
   ```
   `src/config.py` -> `LLM_MODEL` / `OLLAMA_HOST` point at it.
2. **A Piper voice**, downloaded to `models/`:
   ```bash
   python -m piper.download_voices en_US-lessac-medium --data-dir models
   ```
   `src/config.py` -> `PIPER_MODEL_PATH` / `PIPER_CONFIG_PATH` point at the
   downloaded `.onnx` / `.onnx.json` files. Browse other voices at the
   [Piper voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md).
3. **A working microphone/speaker** reachable by `sounddevice` (PortAudio).
   `faster-whisper` downloads its own Whisper model weights on first use.

## Run

```bash
python main.py
```

One window opens: the live video with red bounding boxes on the left, and a
side panel on the right with three sections — a status banner, "Detected
Objects", and "Chat History".

- Press `v` once to start recording a question; the status banner shows a
  blinking red **● Recording** dot (this part is genuinely live). Press `v`
  again to stop and send it.
- While your question is processed, the banner steps through
  **Transcribing... → Thinking... → Speaking...** (orange / yellow /
  green), and the reply is spoken back through your speakers.
- Once a turn completes, "Chat History" shows the tail of the conversation
  (your transcribed question and the assistant's reply), most recent at the
  bottom.
- Press `q` to quit.

Notes:
- The video window freezes for the few seconds it takes to transcribe,
  think, and speak during a voice turn — the vision and voice pipelines
  share one thread in this version. Each status is still shown on screen
  for the whole step it describes, since the panel is redrawn right before
  that step starts.
- A key pressed while a voice turn is mid-flight (e.g. `q` during
  "Thinking...") is not registered — press it again once the turn finishes.

## Configuration

Tunable values (camera index, resolution, model, confidence threshold, box
color, panel width/colors, voice key, Whisper model size, Ollama model/host,
Piper voice paths) live in [src/config.py](src/config.py).
