# eye-ai

Reads a webcam stream, recognizes objects in each frame, draws bounding
boxes and labels, and shows the raw input and annotated output side by side.

This is the first step toward a larger vision assistant: a later phase will
add a local LLM (with TTS/STT) so you can talk to it about what it sees.

## Architecture

Each module owns exactly one concern:

- [src/camera/camera_stream.py](src/camera/camera_stream.py) — `CameraStream`: opens the webcam and returns raw frames. Knows nothing about detection or display.
- [src/detection/object_detector.py](src/detection/object_detector.py) — `ObjectDetector`: takes a frame, runs YOLOv8 detection, draws boxes/labels, returns the annotated frame (and the raw `Detection` list). Knows nothing about the camera or display.
- [src/display/stream_display.py](src/display/stream_display.py) — `StreamDisplay`: shows the input and output frames in two windows and reports quit requests. Knows nothing about capture or detection.
- [main.py](main.py) — wires the three modules together in a simple read → detect → show loop.

Because each module only depends on plain data (numpy frames, `Detection`
dataclasses) rather than on each other, any one of them can be swapped
(e.g. a different camera source, a different model backend, a different
display target) without touching the others.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

The first run downloads the `yolov8n.pt` pretrained weights automatically.

## Run

```bash
python main.py
```

Two windows open: "Input Stream" (raw webcam) and "Output Stream
(Detections)" (annotated). Press `q` in either window to quit.

## Configuration

Tunable values (camera index, resolution, model, confidence threshold,
window names) live in [src/config.py](src/config.py).
