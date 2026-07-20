"""Object description module.

Sole responsibility: given a frame and the detections found in it, produce a
short human-readable description (dominant color, size, an interesting
feature) for each one. Knows nothing about how detections were produced or
how descriptions will be displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.detection.object_detector import Detection

# A short, non-exhaustive set of interesting facts keyed by COCO class name.
# Classes not listed fall back to a generic sentence.
_INTERESTING_FACTS = {
    "person": "Detected as a human subject in frame.",
    "cat": "Cats can rotate their ears roughly 180 degrees.",
    "dog": "Dogs have a sense of smell tens of thousands of times stronger than ours.",
    "chair": "A seat designed for one person, typically with a back and four legs.",
    "laptop": "A portable computer combining screen, keyboard, and battery in one case.",
    "cell phone": "A handheld device typically packing more compute than early spacecraft.",
    "cup": "A small open container, usually for drinking.",
    "bottle": "A narrow-necked container for holding liquid.",
    "book": "A bound set of printed or blank pages.",
    "keyboard": "A standard QWERTY layout is over 140 years old.",
    "mouse": "Optical mice track movement using a tiny built-in camera sensor.",
    "tv": "Modern TVs can pack millions of individually lit pixels.",
    "remote": "A handheld controller, usually infrared or RF based.",
    "backpack": "A bag worn on the back, carried by shoulder straps.",
    "clock": "A device for measuring and displaying time.",
    "vase": "A decorative container, often used to hold flowers.",
    "scissors": "A cutting tool made of two pivoted blades.",
    "teddy bear": "A stuffed toy bear, a design dating back to 1902.",
    "potted plant": "A plant grown in a container rather than in open ground.",
    "couch": "An upholstered seat for more than one person.",
    "dining table": "A table used for meals and gatherings.",
    "bowl": "A round, open container for food or liquid.",
    "banana": "Botanically, a banana is classified as a berry.",
    "apple": "There are over 7,500 known apple cultivars worldwide.",
    "orange": "Oranges are a hybrid between pomelo and mandarin.",
    "umbrella": "A folding canopy designed to protect against rain or sun.",
    "handbag": "A bag with handles, carried in the hand or on the shoulder.",
    "tie": "A strip of fabric worn around the neck, usually for formal wear.",
    "suitcase": "A rigid or semi-rigid case for carrying belongings while traveling.",
    "bicycle": "A human-powered, pedal-driven vehicle with two wheels.",
    "car": "A wheeled motor vehicle typically used for transporting people.",
    "bench": "A long seat for multiple people, often found outdoors.",
    "bed": "A piece of furniture used mainly for sleeping.",
    "sink": "A fixed basin with a water supply, used for washing.",
    "refrigerator": "An appliance that keeps food cold via a compression cycle.",
}
_DEFAULT_FACT = "A common object recognized by the detection model."


@dataclass(frozen=True)
class ObjectDescription:
    label: str
    confidence: float
    color_name: str
    size_label: str
    width_px: int
    height_px: int
    feature_text: str

    @property
    def summary(self) -> str:
        return (
            f"{self.label.capitalize()} ({self.confidence:.0%}) — {self.size_label}, "
            f"{self.width_px}x{self.height_px}px, predominantly {self.color_name}. "
            f"{self.feature_text}"
        )


class ObjectDescriber:
    """Derives a short description for each detection in a frame."""

    def __init__(
        self,
        small_area_ratio: float = 0.05,
        large_area_ratio: float = 0.2,
    ) -> None:
        self._small_area_ratio = small_area_ratio
        self._large_area_ratio = large_area_ratio

    def describe(self, frame: np.ndarray, detections: List[Detection]) -> List[ObjectDescription]:
        frame_area = frame.shape[0] * frame.shape[1]
        return [self._describe_one(frame, detection, frame_area) for detection in detections]

    def _describe_one(self, frame: np.ndarray, detection: Detection, frame_area: int) -> ObjectDescription:
        x1, y1, x2, y2 = detection.box
        crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]

        width_px = max(x2 - x1, 0)
        height_px = max(y2 - y1, 0)
        color_name = self._dominant_color_name(crop)
        size_label = self._size_label(width_px * height_px, frame_area)
        feature_text = _INTERESTING_FACTS.get(detection.label, _DEFAULT_FACT)

        return ObjectDescription(
            label=detection.label,
            confidence=detection.confidence,
            color_name=color_name,
            size_label=size_label,
            width_px=width_px,
            height_px=height_px,
            feature_text=feature_text,
        )

    def _size_label(self, box_area: int, frame_area: int) -> str:
        if frame_area == 0:
            return "unknown size"
        ratio = box_area / frame_area
        if ratio < self._small_area_ratio:
            return "small"
        if ratio < self._large_area_ratio:
            return "medium-sized"
        return "large"

    def _dominant_color_name(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"

        mean_bgr = crop.reshape(-1, 3).mean(axis=0)
        pixel = np.uint8([[mean_bgr]])
        hue, saturation, value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

        if value < 50:
            return "black"
        if saturation < 40 and value > 200:
            return "white"
        if saturation < 40:
            return "gray"

        if hue < 8 or hue >= 172:
            return "red"
        if hue < 20:
            return "orange"
        if hue < 33:
            return "yellow"
        if hue < 78:
            return "green"
        if hue < 100:
            return "cyan"
        if hue < 130:
            return "blue"
        if hue < 155:
            return "purple"
        return "pink"
