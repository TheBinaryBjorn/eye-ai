"""Entry point: wires the camera, detector, and display modules together."""

from __future__ import annotations

from src import config
from src.camera.camera_stream import CameraStream
from src.detection.object_detector import ObjectDetector
from src.display.stream_display import StreamDisplay


def main() -> None:
    detector = ObjectDetector(config.MODEL_NAME, config.CONFIDENCE_THRESHOLD)

    with CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT) as camera, \
            StreamDisplay(config.INPUT_WINDOW_NAME, config.OUTPUT_WINDOW_NAME) as display:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            annotated_frame, _detections = detector.detect(frame)
            display.show(frame, annotated_frame)

            if display.should_quit(config.QUIT_KEY):
                break


if __name__ == "__main__":
    main()
