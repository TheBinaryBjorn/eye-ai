"""Entry point: wires the camera, detector, describer, and display modules together."""

from __future__ import annotations

from src import config
from src.camera.camera_stream import CameraStream
from src.description.object_describer import ObjectDescriber
from src.detection.object_detector import ObjectDetector
from src.display.stream_display import StreamDisplay


def main() -> None:
    detector = ObjectDetector(config.MODEL_NAME, config.CONFIDENCE_THRESHOLD, config.BOX_COLOR)
    describer = ObjectDescriber()

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
            descriptions = describer.describe(frame, detections)
            display.show(annotated_frame, descriptions)

            if display.should_quit(config.QUIT_KEY):
                break


if __name__ == "__main__":
    main()
