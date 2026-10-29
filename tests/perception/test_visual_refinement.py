"""Screenshot-only close-up recovery for targets missed by the first scan."""

import io
import unittest
from types import SimpleNamespace

from PIL import Image

from yewo.agent.run.control_discovery import ControlDiscoveryTransaction


def _pending(phash, region, intent="activate", anchor_above=None, anchor_below=None):
    return ControlDiscoveryTransaction(
        source_phash=phash,
        region=region,
        description="the control",
        intent=intent,
        anchor_above=anchor_above,
        anchor_below=anchor_below,
    )


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

    def test_missing_anchors_refuse_a_fixed_screen_third(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.step.observation import number_targets

        descriptor = SimpleNamespace(describe=lambda _png: self.fail("must not scan an unbounded crop"))
        perception = SimpleNamespace(image_descriptor=descriptor)
        ctx = SimpleNamespace(
            control_discovery=_pending("screen-1", "lower_center"),
            keyboard_open=False,
            screen_height=1800,
            refused_marks={},
        )
        snapshot = self._snapshot()

        number_targets(snapshot, ctx, GROUNDING_MARKS, perception)

        self.assertEqual(snapshot.marks, [])
        self.assertIsNone(snapshot.refinement)
        self.assertIn("could not be bounded", snapshot.intervention_failure_reason)
        self.assertIsNotNone(ctx.control_discovery)

    def test_a_request_for_an_old_screen_is_not_refined(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.step.observation import number_targets

        descriptor = SimpleNamespace(describe=lambda _png: self.fail("must not scan a stale crop"))
        ctx = SimpleNamespace(
            control_discovery=_pending("old-screen", "lower_center"),
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

    def test_minor_capture_jitter_does_not_make_the_request_stale(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.step.observation import number_targets

        snapshot = self._snapshot()
        snapshot.current_phash = "0000000000000001"
        ctx = SimpleNamespace(
            control_discovery=_pending("0000000000000000", "lower_center"),
            keyboard_open=False,
            screen_height=1800,
            refused_marks={},
        )

        number_targets(
            snapshot,
            ctx,
            GROUNDING_MARKS,
            SimpleNamespace(image_descriptor=None),
        )

        self.assertIn("could not be bounded", snapshot.intervention_failure_reason)


class DerivedRegionTests(unittest.TestCase):
    """Surrounding controls replace the third-of-screen guess where they exist."""

    def _element(self, box, text=""):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def _form(self):
        return [
            self._element((42, 1100, 480, 1162), "Repeations:"),
            self._element((42, 1280, 420, 1320), "Pause [s]:"),
            self._element((42, 950, 1038, 1010)),
            self._element((42, 1340, 1038, 1400)),
        ]

    def _snapshot(self, elements):
        from yewo.agent.step.observation import PerceptionSnapshot

        image = Image.new("RGB", (1080, 2400), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return PerceptionSnapshot(
            gui_elements=elements,
            marks=[],
            raw_img_bytes=buffer.getvalue(),
            planner_img_bytes=buffer.getvalue(),
            annotated_img=image,
            current_phash="screen-1",
            timestamp="20261025_120000",
        )

    def _refine(self, elements, detected, intent="enter_text"):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.step.observation import number_targets

        descriptor = SimpleNamespace(
            describe=lambda _png: (detected, Image.new("RGB", (1, 1)))
        )
        ctx = SimpleNamespace(
            control_discovery=_pending(
                "screen-1",
                "middle_left",
                intent,
                anchor_above=(42, 1100, 480, 1162),
                anchor_below=(42, 1280, 420, 1320),
            ),
            keyboard_open=False,
            screen_height=2400,
            refused_marks={},
        )
        snapshot = self._snapshot(elements)
        number_targets(
            snapshot, ctx, GROUNDING_MARKS, SimpleNamespace(image_descriptor=descriptor)
        )
        return snapshot

    def test_the_crop_is_bounded_by_the_labels_around_the_gap(self):
        snapshot = self._refine(self._form(), [])

        self.assertEqual(snapshot.refinement["crop"], [36, 1156, 1044, 1286])

    def test_the_crop_records_which_controls_bounded_it(self):
        snapshot = self._refine(self._form(), [])

        derivation = snapshot.refinement["derivation"]
        self.assertEqual(derivation["anchor_above"], [42, 1100, 480, 1162])
        self.assertEqual(derivation["anchor_below"], [42, 1280, 420, 1320])

    def test_a_derived_band_is_magnified_further_than_a_third_of_screen(self):
        snapshot = self._refine(self._form(), [])

        self.assertGreater(snapshot.refinement["scale"], 2)

    def test_a_region_that_numbers_nothing_offers_cells_to_try(self):
        snapshot = self._refine(self._form(), [])

        self.assertIsNotNone(snapshot.intervention)
        self.assertEqual(snapshot.intervention.intent, "enter_text")
        self.assertTrue(snapshot.intervention.candidates)

    def test_a_region_that_numbers_something_offers_no_cells(self):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        found = GUIElement(
            element="EditText",
            type="yolo_detection",
            confidence=0.9,
            bounding_box=BoundingBox(x_min=20, y_min=20, x_max=400, y_max=200),
        )

        snapshot = self._refine(self._form(), [found])

        self.assertIsNone(snapshot.intervention)
        self.assertTrue(snapshot.marks)
