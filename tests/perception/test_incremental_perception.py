import unittest

from PIL import Image, ImageDraw

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.incremental import (
    merge_ocr_evidence,
    plan_incremental_refresh,
    rescale_ocr_result,
)
from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming


def _element(box, *roles):
    return GUIElement(
        element="ModelClass",
        type="yolo_detection",
        confidence=0.9,
        semantic_roles=frozenset(roles),
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def _line(text, box):
    left, top, right, bottom = box
    return OCRLine(
        box=(
            OCRPoint(x=left, y=top),
            OCRPoint(x=right, y=top),
            OCRPoint(x=right, y=bottom),
            OCRPoint(x=left, y=bottom),
        ),
        text=text,
        confidence=0.9,
    )


class IncrementalPlanningTests(unittest.TestCase):
    def _changed_image(self, box):
        image = Image.new("RGB", (400, 400), "white")
        changed = image.copy()
        ImageDraw.Draw(changed).rectangle(box, fill="black")
        return image, changed

    def test_small_change_inside_keyboard_region_reuses_detector(self):
        previous, current = self._changed_image((150, 150, 165, 165))
        elements = [
            _element((100, 100, 300, 300), "keyboard_region"),
            _element((140, 140, 180, 180), "keyboard_key"),
        ]

        plan = plan_incremental_refresh(previous, current, elements, frozenset({"keyboard_region"}))

        self.assertTrue(plan.reuse_detector)
        self.assertEqual(plan.surface_roles, frozenset({"keyboard_region"}))
        self.assertLess(plan.ocr_input_pixels, 400 * 400)

    def test_change_at_semantic_surface_boundary_runs_detector(self):
        previous, current = self._changed_image((100, 150, 110, 160))
        region = _element((100, 100, 300, 300), "keyboard_region")

        plan = plan_incremental_refresh(previous, current, [region], frozenset({"keyboard_region"}))

        self.assertFalse(plan.reuse_detector)

    def test_unknown_future_role_does_not_authorize_detector_reuse(self):
        previous, current = self._changed_image((150, 150, 165, 165))
        region = _element((100, 100, 300, 300), "future_region")

        plan = plan_incremental_refresh(previous, current, [region], frozenset({"keyboard_region"}))

        self.assertFalse(plan.reuse_detector)

    def test_large_change_falls_back_to_full_scan(self):
        previous, current = self._changed_image((0, 0, 399, 100))

        plan = plan_incremental_refresh(previous, current, [], frozenset())

        self.assertEqual(plan.full_scan_reason, "changed_area_too_large")


class IncrementalOCRTests(unittest.TestCase):
    def test_calibrated_crop_coordinates_return_to_native_pixels(self):
        result = OCRResult(
            engine_identity="rapidocr@test",
            lines=(_line("scaled", (5, 10, 25, 20)),),
            timing=OCRTiming(engine_total_ms=1, wall_ms=1),
        )

        restored = rescale_ocr_result(result, 2.0)

        self.assertEqual(restored.lines[0].box[0], OCRPoint(x=10, y=20))
        self.assertEqual(restored.lines[0].box[2], OCRPoint(x=50, y=40))

    def test_retains_untouched_lines_and_maps_crop_lines_to_screen(self):
        previous = (_line("keep", (10, 10, 40, 30)), _line("old", (210, 210, 250, 230)))
        crop_result = OCRResult(
            engine_identity="rapidocr@test",
            lines=(_line("new", (10, 10, 50, 30)),),
            timing=OCRTiming(engine_total_ms=2, wall_ms=3),
        )

        merged = merge_ocr_evidence(
            previous,
            [((200, 200, 300, 260), crop_result)],
            "rapidocr@test",
        )

        self.assertEqual([line.text for line in merged.lines], ["keep", "new"])
        self.assertEqual(merged.lines[1].box[0], OCRPoint(x=210, y=210))
        self.assertEqual(merged.timing.wall_ms, 3)


if __name__ == "__main__":
    unittest.main()
