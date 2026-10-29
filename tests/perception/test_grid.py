"""Candidate cells over a region nothing was detected in."""

import unittest

CROP = (36, 1156, 1044, 1286)


def _el(box, confidence=0.9):
    from yewo.agent.perception.elements import BoundingBox, GUIElement

    return GUIElement(
        element="View",
        type="yolo_detection",
        confidence=confidence,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


class GridTests(unittest.TestCase):
    def test_cells_never_outnumber_the_probe_budget(self):
        from yewo.agent.grounding.grid import MAX_CANDIDATES, build

        self.assertLessEqual(len(build((0, 0, 1080, 2400))), MAX_CANDIDATES)

    def test_every_cell_is_at_least_a_touch_target(self):
        from yewo.agent.grounding.grid import MIN_CELL_PIXELS, build

        for candidate in build(CROP):
            width = candidate.box[2] - candidate.box[0]
            height = candidate.box[3] - candidate.box[1]
            with self.subTest(cell=candidate.index):
                self.assertGreaterEqual(max(width, height), MIN_CELL_PIXELS)

    def test_a_region_smaller_than_one_cell_still_offers_one(self):
        from yewo.agent.grounding.grid import build

        self.assertEqual(len(build((0, 0, 20, 20))), 1)

    def test_the_cells_cover_the_region_and_nothing_outside_it(self):
        from yewo.agent.grounding.grid import build

        boxes = [candidate.box for candidate in build(CROP)]

        self.assertEqual(min(box[0] for box in boxes), CROP[0])
        self.assertEqual(min(box[1] for box in boxes), CROP[1])
        self.assertEqual(max(box[2] for box in boxes), CROP[2])
        self.assertEqual(max(box[3] for box in boxes), CROP[3])

    def test_the_first_cell_tried_is_the_middle_of_the_region(self):
        from yewo.agent.grounding.grid import build

        first = build(CROP)[0]
        centre = (CROP[0] + CROP[2]) // 2, (CROP[1] + CROP[3]) // 2

        self.assertLessEqual(first.box[0], centre[0])
        self.assertGreaterEqual(first.box[2], centre[0])
        self.assertLessEqual(first.box[1], centre[1])
        self.assertGreaterEqual(first.box[3], centre[1])

    def test_cells_are_numbered_in_the_order_they_are_tried(self):
        from yewo.agent.grounding.grid import build

        indexes = [candidate.index for candidate in build(CROP)]

        self.assertEqual(indexes, list(range(1, len(indexes) + 1)))

    def test_a_cell_over_a_detection_carries_it(self):
        from yewo.agent.grounding.grid import build

        covering = _el((36, 1156, 1044, 1286))

        self.assertTrue(all(c.detection is covering for c in build(CROP, [covering])))

    def test_a_cell_over_nothing_carries_no_detection(self):
        from yewo.agent.grounding.grid import build

        self.assertTrue(all(c.detection is None for c in build(CROP, [])))

    def test_capacity_reports_the_cells_the_budget_leaves_out(self):
        from yewo.agent.grounding.grid import MAX_CANDIDATES, build, cell_capacity

        crop = (0, 0, 1080, 2400)

        self.assertGreater(cell_capacity(crop), MAX_CANDIDATES)
        self.assertGreater(cell_capacity(crop), len(build(crop)))

    def test_the_overlay_draws_on_the_magnified_crop_it_was_given(self):
        from PIL import Image

        from yewo.agent.grounding.grid import build, overlay

        scale = 3
        size = ((CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale)
        close_up = Image.new("RGB", size, "white")

        drawn = overlay(close_up, build(CROP), CROP, scale)

        self.assertEqual(Image.open(__import__("io").BytesIO(drawn)).size, size)
