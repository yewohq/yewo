import hashlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.fusion import fuse_perception
from yewo.agent.perception.ocr import OCRResult, OCRTiming
from yewo.agent.perception.screen import ScreenPerception


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeDescriptor:
    model_identity = "fake-detector"

    def __init__(self, elements):
        self.elements = elements
        self.incremental_elements = None
        self.last_frame = None
        self.full_calls = 0
        self.incremental_calls = []

    def _frame(self, png_bytes):
        digest = hashlib.sha256(png_bytes).hexdigest()
        result = OCRResult(
            engine_identity="fake-ocr",
            lines=(),
            timing=OCRTiming(engine_total_ms=0, wall_ms=0),
        )
        self.last_frame = fuse_perception(self.elements, result, digest, self.model_identity)

    def describe(self, png_bytes):
        self.full_calls += 1
        self._frame(png_bytes)
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return [element.model_copy(deep=True) for element in self.elements], image

    def describe_incremental(self, png_bytes, previous_frame, regions, reuse_detector):
        self.incremental_calls.append((regions, reuse_detector, previous_frame))
        returned = self.incremental_elements
        if returned is None:
            returned = self.elements
        original = self.elements
        self.elements = returned
        self._frame(png_bytes)
        self.elements = original
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return [element.model_copy(deep=True) for element in returned], image

    def clear_cache(self):
        self.last_frame = None


class FakeHash:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        return abs(self.value - other.value)

    def __str__(self):
        return f"{self.value:064x}"


class ScreenIncrementalTests(unittest.TestCase):
    def _perception(self, images):
        region = GUIElement(
            element="Surface",
            type="yolo_detection",
            confidence=0.9,
            semantic_roles=frozenset({"keyboard_region"}),
            bounding_box=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=300),
        )
        key = GUIElement(
            element="Key",
            type="yolo_detection",
            confidence=0.9,
            semantic_roles=frozenset({"keyboard_key"}),
            bounding_box=BoundingBox(x_min=140, y_min=140, x_max=180, y_max=180),
        )
        screenshots = iter(_png(image) for image in images)
        perception = ScreenPerception.__new__(ScreenPerception)
        perception.device_session = SimpleNamespace(
            device_control=SimpleNamespace(get_screenshot=lambda: next(screenshots))
        )
        perception.image_descriptor = FakeDescriptor([region, key])
        perception._last_hash = None
        perception._last_screenshot = None
        perception._last_elements = None
        perception._last_frame = None
        perception.last_scan_info = None
        return perception

    def _perceive_twice(self, perception):
        with patch(
            "yewo.agent.perception.screen.imagehash.phash",
            side_effect=(FakeHash(1), FakeHash(2)),
        ):
            perception.perceive()
            perception.perceive()

    def test_small_role_contained_change_reuses_detector_and_partial_ocr(self):
        first = Image.new("RGB", (400, 400), "white")
        second = first.copy()
        ImageDraw.Draw(second).rectangle((150, 150, 165, 165), fill="black")
        perception = self._perception([first, second])

        self._perceive_twice(perception)

        descriptor = perception.image_descriptor
        self.assertEqual(descriptor.full_calls, 1)
        self.assertEqual(len(descriptor.incremental_calls), 1)
        self.assertTrue(descriptor.incremental_calls[0][1])
        self.assertTrue(perception.last_scan_info.detector_reused)
        self.assertLess(perception.last_scan_info.ocr_input_pixels, 400 * 400)

    def test_large_change_uses_full_pipeline(self):
        first = Image.new("RGB", (400, 400), "white")
        second = first.copy()
        ImageDraw.Draw(second).rectangle((0, 0, 399, 100), fill="black")
        perception = self._perception([first, second])

        self._perceive_twice(perception)

        descriptor = perception.image_descriptor
        self.assertEqual(descriptor.full_calls, 2)
        self.assertEqual(descriptor.incremental_calls, [])
        self.assertEqual(
            perception.last_scan_info.full_scan_fallback,
            "changed_area_too_large",
        )

    def test_unproven_detector_change_falls_back_to_full_frame(self):
        first = Image.new("RGB", (400, 400), "white")
        second = first.copy()
        ImageDraw.Draw(second).rectangle((100, 150, 110, 160), fill="black")
        perception = self._perception([first, second])
        perception.image_descriptor.incremental_elements = []

        self._perceive_twice(perception)

        descriptor = perception.image_descriptor
        self.assertEqual(len(descriptor.incremental_calls), 1)
        self.assertFalse(descriptor.incremental_calls[0][1])
        self.assertEqual(descriptor.full_calls, 2)
        self.assertEqual(
            perception.last_scan_info.full_scan_fallback,
            "detector_structure_changed",
        )


if __name__ == "__main__":
    unittest.main()
