"""Screenshot-only close-up recovery for targets missed by the first scan."""

import io
import unittest
from types import SimpleNamespace

from PIL import Image


class VisualRefinementTests(unittest.TestCase):
    def _snapshot(self):
        from yewo.agent.step.observation import PerceptionSnapshot

        image = Image.new("RGB", (900, 1800), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return PerceptionSnapshot(
            gui_elements=[],
            marks=[],
            raw_img_bytes=buffer.getvalue(),
            planner_img_bytes=buffer.getvalue(),
            annotated_img=image,
            current_phash="screen-1",
            timestamp="20261025_120000",
        )

    def test_a_crop_detection_maps_back_to_full_screen_coordinates(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.step.observation import number_targets

        element = GUIElement(
            element="Button",
            type="yolo_detection",
            confidence=0.9,
            text="MOVE",
            bounding_box=BoundingBox(x_min=200, y_min=300, x_max=400, y_max=500),
        )
        descriptor = SimpleNamespace(describe=lambda _png: ([element], Image.new("RGB", (1, 1))))
        perception = SimpleNamespace(image_descriptor=descriptor)
        ctx = SimpleNamespace(
            pending_visual_refinement=("screen-1", "lower_center"),
            keyboard_open=False,
            screen_height=1800,
            refused_marks={},
        )
        snapshot = self._snapshot()

        number_targets(snapshot, ctx, GROUNDING_MARKS, perception)

        self.assertEqual(tuple(snapshot.marks[0].element.bounding_box), (325, 1200, 425, 1300))
        self.assertEqual(snapshot.marks[0].center, (375, 1250))
        self.assertEqual(Image.open(io.BytesIO(snapshot.planner_img_bytes)).size, (900, 1500))
        self.assertEqual(snapshot.refinement["crop"], [225, 1050, 675, 1800])
        self.assertIsNone(ctx.pending_visual_refinement)

    def test_a_request_for_an_old_screen_is_not_refined(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.step.observation import number_targets

        descriptor = SimpleNamespace(describe=lambda _png: self.fail("must not scan a stale crop"))
        ctx = SimpleNamespace(
            pending_visual_refinement=("old-screen", "lower_center"),
            keyboard_open=False,
            screen_height=1800,
            refused_marks={},
        )
        snapshot = self._snapshot()

        number_targets(
            snapshot,
            ctx,
            GROUNDING_MARKS,
            SimpleNamespace(image_descriptor=descriptor),
        )

        self.assertIsNone(snapshot.refinement)
        self.assertEqual(snapshot.marks, [])
