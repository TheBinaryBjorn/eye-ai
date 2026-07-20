# eye-ai

Reads a webcam stream, recognizes objects in each frame, draws red bounding
boxes around them, and shows a single window with the annotated video on
the left and a description panel (color, size, an interesting fact per
object) on the right.

This is the first step toward a larger vision assistant: a later phase will
add a local LLM (with TTS/STT) so you can talk to it about what it sees.

## Architecture

Each module owns exactly one concern:

- [src/camera/camera_stream.py](src/camera/camera_stream.py) — `CameraStream`: opens the webcam and returns raw frames. Knows nothing about detection, description, or display.
- [src/detection/object_detector.py](src/detection/object_detector.py) — `ObjectDetector`: takes a frame, runs YOLOv8 detection, draws red boxes/labels, returns the annotated frame and the raw `Detection` list (label, confidence, box). Knows nothing about the camera, descriptions, or display.
- [src/description/object_describer.py](src/description/object_describer.py) — `ObjectDescriber`: takes the original frame and `Detection` list, and for each one derives a dominant color, a size classification, and a short fact, returning `ObjectDescription` records. Knows nothing about the camera, detection internals, or display.
- [src/display/stream_display.py](src/display/stream_display.py) — `StreamDisplay`: composites the annotated frame and the `ObjectDescription` list into one window (video + side panel) and reports quit requests. Knows nothing about capture, detection, or description logic.
- [main.py](main.py) — wires the four modules together in a read → detect → describe → show loop.

Because each module only depends on plain data (numpy frames, `Detection`
and `ObjectDescription` dataclasses) rather than on each other's internals,
any one of them can be swapped (e.g. a different camera source, a different
model backend, richer descriptions, a different display target) without
touching the others.

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

One window opens: the live video with red bounding boxes on the left, and a
"Detected Objects" panel on the right listing each object's name,
confidence, size, dominant color, and a short fact. Press `q` to quit.

## Configuration

Tunable values (camera index, resolution, model, confidence threshold, box
color, panel width/colors) live in [src/config.py](src/config.py).
