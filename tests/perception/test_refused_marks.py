"""A number proven to hold no text is not offered for text again."""

import unittest
from types import SimpleNamespace


class RefusedMarkTests(unittest.TestCase):
    """A number that took no text is not offered again on that screen.

    Nothing visible separates a field from a picture of one. The only evidence
    is having typed there and watched the cursor stay put, so that evidence has
    to survive the step it was gathered in.
    """

    def _marks(self, refuses_text=False):
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.grounding.marks import Mark

        element = GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text="",
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=30, y_max=40),
        )
        return [Mark(index=17, element=element, refuses_text=refuses_text)]

    def _resolve(self, action, refuses_text):
        from yewo.agent.plan.planner import GROUNDING_MARKS
        from yewo.agent.step.targeting import resolve_mark

        return resolve_mark(action, self._marks(refuses_text), GROUNDING_MARKS)

    def _type(self):
        from yewo.agent.actuation.actions import TypeMarkAction

        return TypeMarkAction(
            mark=17,
            target="Card number field",
            text="4111222233334444",
            reason="card",
        )

    def _tap(self):
        from yewo.agent.actuation.actions import TapMarkAction

        return TapMarkAction(mark=17, target="Card number field", reason="open it")

    def test_typing_into_a_refused_mark_never_reaches_the_device(self):
        unresolved, resolved = self._resolve(self._type(), refuses_text=True)

        self.assertIsNone(resolved)
        self.assertIn("holds no text", unresolved)
        self.assertIn("17", unresolved)

    def test_the_refusal_says_what_to_do_instead(self):
        """Naming the fault without a next move is what kept the loop going."""
        unresolved, _ = self._resolve(self._type(), refuses_text=True)

        self.assertIn("Choose a different number", unresolved)
        self.assertIn("platform driver", unresolved)

    def test_a_mark_that_was_never_tried_still_resolves(self):
        unresolved, resolved = self._resolve(self._type(), refuses_text=False)

        self.assertIsNone(unresolved)
        self.assertEqual((resolved.coordinates.x, resolved.coordinates.y), (20, 30))

    def test_tapping_a_refused_mark_is_still_allowed(self):
        """It holds no text. It may still be the way to reveal what does."""
        unresolved, resolved = self._resolve(self._tap(), refuses_text=True)

        self.assertIsNone(unresolved)
        self.assertIsNotNone(resolved)

    def test_the_planner_is_told_which_numbers_hold_no_text(self):
        from yewo.agent.grounding.marks import describe

        listing = describe(self._marks(refuses_text=True), height=2400)

        self.assertIn("17", listing)
        self.assertIn("holds no text", listing)
        self.assertNotIn("holds no text", describe(self._marks(), height=2400))


class RefusalRecordingTests(unittest.TestCase):
    """Learning the fact, and carrying it to the next step's numbering."""

    def _ctx(self):
        from yewo.agent.run.context import RunContext

        return RunContext(history=None, loop_detector=None)

    def _marks_at(self, *boxes):
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.grounding.marks import Mark

        return [
            Mark(
                index=n,
                element=GUIElement(
                    element="Text",
                    type="yolo_detection",
                    confidence=0.9,
                    text="",
                    bounding_box=BoundingBox(x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3]),
                ),
            )
            for n, b in enumerate(boxes, 1)
        ]

    def test_one_failure_is_not_yet_proof(self):
        """A tap into the padding of a real field also leaves focus unmoved."""
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        marks = self._marks_at((0, 0, 9, 9), (10, 20, 30, 40))
        refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 2, marks)

        self.assertEqual(ctx.refused_marks, {})
        self.assertEqual(ctx.text_strikes, {"screenA": [(10, 20, 30, 40)]})

    def test_the_place_is_banned_on_the_second_failure(self):
        """A number is reading order over detections and does not survive them."""
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        marks = self._marks_at((0, 0, 9, 9), (10, 20, 30, 40))
        refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 2, marks)
        refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 2, marks)

        self.assertEqual(ctx.refused_marks, {"screenA": [(10, 20, 30, 40)]})

    def test_a_second_failure_elsewhere_bans_neither(self):
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        marks = self._marks_at((0, 0, 9, 9), (10, 20, 30, 40))
        refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 1, marks)
        refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 2, marks)

        self.assertEqual(ctx.refused_marks, {})

    def test_a_refusal_is_scoped_to_the_screen_it_happened_on(self):
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        marks = self._marks_at((10, 20, 30, 40))
        for _ in range(2):
            refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenA", 1, marks)
            refuse_mark(ctx.text_strikes, ctx.refused_marks, "screenB", 1, marks)

        self.assertEqual(set(ctx.refused_marks), {"screenA", "screenB"})

    def test_an_unidentified_screen_records_nothing(self):
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        refuse_mark(ctx.text_strikes, ctx.refused_marks, None, 1, self._marks_at((10, 20, 30, 40)))

        self.assertEqual(ctx.refused_marks, {})
        self.assertEqual(ctx.text_strikes, {})

    def test_a_number_naming_nothing_records_nothing(self):
        """The planner can answer with a number the rebuild no longer has."""
        from yewo.agent.grounding.marks import refuse_mark

        ctx = self._ctx()
        refuse_mark(
            ctx.text_strikes, ctx.refused_marks, "screenA", 99, self._marks_at((10, 20, 30, 40))
        )

        self.assertEqual(ctx.refused_marks, {})
        self.assertEqual(ctx.text_strikes, {})

    def test_focus_that_never_moved_raises_the_type_the_loop_records_on(self):
        """Any other rejection must not burn a number."""
        from yewo.errors import ActionRejected, TargetNotInteractive
        from yewo.agent.device.focus import Focus

        rect = {"x": 63, "y": 820, "width": 954, "height": 147}
        focus = Focus(SimpleNamespace(), screen_area=1080 * 2400)
        focus._rect = lambda: rect

        with self.assertRaises(TargetNotInteractive):
            focus.assert_at(537, 1260, "TYPE", before=rect)

        with self.assertRaises(ActionRejected) as moved:
            focus.assert_at(537, 1260, "TYPE", before={"x": 0, "y": 0, "width": 1, "height": 1})
        self.assertNotIsInstance(moved.exception, TargetNotInteractive)

    def test_what_was_learned_is_applied_to_the_next_numbering(self):
        """The numbers are rebuilt every step, so the flag has to be re-applied.

        This is the link between learning the fact and acting on it. Drop it and
        the guard still works and the prompt still renders, but no mark is ever
        flagged, so nothing changes.
        """
        import io

        from PIL import Image

        from yewo.agent.plan.planner import GROUNDING_MARKS
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.step.observation import number_targets

        png = io.BytesIO()
        Image.new("RGB", (1080, 2400), "white").save(png, format="PNG")

        def element(y):
            return GUIElement(
                element="Text",
                type="yolo_detection",
                confidence=0.9,
                text="",
                bounding_box=BoundingBox(x_min=10, y_min=y, x_max=200, y_max=y + 40),
            )

        snapshot = SimpleNamespace(
            gui_elements=[element(100), element(300)],
            raw_img_bytes=png.getvalue(),
            planner_img_bytes=png.getvalue(),
            current_phash="screenA",
            marks=[],
        )
        ctx = self._ctx()
        ctx.keyboard_open = False
        ctx.screen_height = 2400
        ctx.refused_marks = {"screenA": [(10, 300, 200, 340)]}

        number_targets(snapshot, ctx, GROUNDING_MARKS)

        flagged = {m.index: m.refuses_text for m in snapshot.marks}
        self.assertEqual(flagged, {1: False, 2: True})

    def test_a_refusal_stays_on_its_element_when_the_screen_renumbers(self):
        """Numbers are reading order over detections, so detections move them.

        A detection appearing or vanishing shifts every number below it while
        the screen fingerprint holds. Remembering a number would then refuse a
        different, working field — the one the recovery exists to reach.
        """
        import io

        from PIL import Image

        from yewo.agent.plan.planner import GROUNDING_MARKS
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.step.observation import number_targets

        png = io.BytesIO()
        Image.new("RGB", (1080, 2400), "white").save(png, format="PNG")

        def element(y):
            return GUIElement(
                element="Text",
                type="yolo_detection",
                confidence=0.9,
                text="",
                bounding_box=BoundingBox(x_min=10, y_min=y, x_max=200, y_max=y + 40),
            )

        def snapshot(elements):
            return SimpleNamespace(
                gui_elements=elements,
                raw_img_bytes=png.getvalue(),
                planner_img_bytes=png.getvalue(),
                current_phash="screenA",
                marks=[],
            )

        ctx = self._ctx()
        ctx.keyboard_open, ctx.screen_height = False, 2400
        # The card at y=300 took no text. It was number 2 at the time.
        ctx.refused_marks = {"screenA": [(10, 300, 200, 340)]}

        # Next pass finds one more element above it, so it is now number 3.
        later = snapshot([element(50), element(100), element(300), element(500)])
        number_targets(later, ctx, GROUNDING_MARKS)

        by_top = {m.element.bounding_box.y_min: m.refuses_text for m in later.marks}
        self.assertTrue(by_top[300], "the refused element lost its refusal")
        self.assertFalse(any(v for k, v in by_top.items() if k != 300))

    def test_a_refusal_does_not_follow_the_agent_to_another_screen(self):
        import io

        from PIL import Image

        from yewo.agent.plan.planner import GROUNDING_MARKS
        from yewo.agent.perception.elements import BoundingBox, GUIElement
        from yewo.agent.step.observation import number_targets

        png = io.BytesIO()
        Image.new("RGB", (1080, 2400), "white").save(png, format="PNG")
        snapshot = SimpleNamespace(
            gui_elements=[
                GUIElement(
                    element="Text",
                    type="yolo_detection",
                    confidence=0.9,
                    text="",
                    bounding_box=BoundingBox(x_min=10, y_min=100, x_max=200, y_max=140),
                )
            ],
            raw_img_bytes=png.getvalue(),
            planner_img_bytes=png.getvalue(),
            current_phash="screenB",
            marks=[],
        )
        ctx = self._ctx()
        ctx.keyboard_open = False
        ctx.screen_height = 2400
        ctx.refused_marks = {"screenA": [(10, 100, 200, 140)]}

        number_targets(snapshot, ctx, GROUNDING_MARKS)

        self.assertFalse(snapshot.marks[0].refuses_text)


class ElementIdentityTests(unittest.TestCase):
    """Recognising one element across two looks at the same screen."""

    def _element(self, box):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text="",
            bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        )

    def test_the_same_box_is_the_same_element(self):
        from yewo.agent.grounding.marks import is_same_element

        self.assertTrue(is_same_element(self._element((10, 300, 200, 340)), (10, 300, 200, 340)))

    def test_a_detector_nudging_the_box_is_still_the_same_element(self):
        """Detections wobble by a pixel or two between passes."""
        from yewo.agent.grounding.marks import is_same_element

        self.assertTrue(is_same_element(self._element((11, 301, 201, 341)), (10, 300, 200, 340)))

    def test_a_neighbour_that_merely_touches_is_a_different_element(self):
        """A stacked form puts real fields right against each other.

        Matching loosely here would refuse typing into the field below the one
        that actually took no text.
        """
        from yewo.agent.grounding.marks import is_same_element

        self.assertFalse(is_same_element(self._element((10, 330, 200, 370)), (10, 300, 200, 340)))

    def test_an_unrelated_box_is_a_different_element(self):
        from yewo.agent.grounding.marks import is_same_element

        self.assertFalse(is_same_element(self._element((10, 900, 200, 940)), (10, 300, 200, 340)))
