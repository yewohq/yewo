"""Deciding whether a named target is safe to touch."""

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fakes


from yewo.errors import ObservationUnavailable


class StalenessTests(unittest.TestCase):
    """Seconds pass between reading the screen and tapping it."""

    def _png(self, fill, patch=None):
        screen = fakes.solid_png(fill)
        if patch is None:
            return screen
        box, colour = patch
        return self._with_patch(screen, box, colour)

    def _with_patch(self, png, box, colour):

        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                image.putpixel((x, y), colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _element(self, box=(10, 10, 60, 60)):
        return fakes.element(box=box)

    def _tap(self, x, y):
        from yewo.agent.actuation.actions import TapAction, Coordinates

        return TapAction(coordinates=Coordinates(x=x, y=y), reason="tap it")

    def test_an_unchanged_target_is_not_stale(self):
        from yewo.agent.step.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))

        self.assertFalse(target_moved(before, before, self._tap(30, 30), [self._element()]))

    def test_a_target_that_moved_is_stale(self):
        from yewo.agent.step.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("white")

        self.assertTrue(target_moved(before, after, self._tap(30, 30), [self._element()]))

    def test_change_elsewhere_on_screen_is_ignored(self):
        """A banner landing in the corner must not throw away a good plan."""
        from yewo.agent.step.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._with_patch(after, (100, 300, 190, 390), (255, 0, 0))

        self.assertFalse(target_moved(before, after, self._tap(30, 30), [self._element()]))

    def test_a_point_outside_every_element_still_gets_checked(self):
        from yewo.agent.step.staleness import target_moved

        before = self._png("white", ((100, 300, 190, 390), (0, 0, 0)))
        after = self._png("white")

        self.assertTrue(target_moved(before, after, self._tap(145, 345), []))

    def test_an_action_with_no_target_is_never_stale(self):
        from yewo.agent.actuation.actions import KeyEventAction
        from yewo.agent.step.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("black")

        self.assertFalse(target_moved(before, after, KeyEventAction(key=4, reason="back"), []))

    def test_an_unreadable_screenshot_does_not_strand_the_run(self):
        from yewo.agent.step.staleness import target_moved

        self.assertFalse(target_moved(b"not a png", b"nor this", self._tap(30, 30), []))


class TargetDecisionTests(unittest.TestCase):
    """Nothing guesses: an unusable target stops the step and is recorded."""

    def _snapshot(self, marks=(), png=None):
        return fakes.snapshot(marks=marks, png=png)

    def _device(self, png=None):
        """Re-reads the same screen the snapshot was taken from, so nothing is stale."""
        screen = fakes.solid_png() if png is None else png
        return SimpleNamespace(get_screenshot=Mock(return_value=screen))

    def _marks(self):
        return fakes.numbered(fakes.element(box=(10, 20, 30, 40)))

    def test_a_valid_number_becomes_the_point_to_touch(self):
        """The number is swapped for its coordinate twin — every mode ends in a point."""
        from yewo.agent.actuation.actions import TapAction, TapMarkAction
        from yewo.agent.step.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, target="Continue button", reason="continue"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertTrue(decision.ok)
        self.assertIsInstance(decision.action, TapAction)
        self.assertEqual((decision.action.coordinates.x, decision.action.coordinates.y), (20, 30))

    def test_a_number_that_names_nothing_stops_the_step(self):
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.step.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=47, target="Continue button", reason="continue"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertFalse(decision.ok)
        self.assertIn("no element 47", decision.observation)
        self.assertFalse(decision.unmatched)

    def test_an_undetected_target_is_reported_without_ending_the_scenario(self):
        """Only what is on screen is numbered, so looking again may find it.

        A field simply behind the keyboard is found by looking again. Whether
        a second refusal ends the run is the loop's call, because only the loop
        knows if the screen changed.
        """
        from yewo.agent.actuation.actions import TargetNotListedAction
        from yewo.agent.step.targeting import evaluate

        decision = evaluate(
            TargetNotListedAction(description="blue button at the bottom", reason="r"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertFalse(decision.ok)
        self.assertTrue(decision.unmatched)
        self.assertIn("blue button at the bottom", decision.observation)
        self.assertIn("Scroll it into view", decision.observation)

    def test_a_missing_target_names_itself_in_the_reason(self):
        """The report should say which wall the run hit, not only that it stopped."""
        from yewo.agent.step.targeting import Decision, stalled_reason

        reason = stalled_reason(Decision(ok=False, unmatched=True, target="the CVV field"))

        self.assertIn("already searched", reason)
        self.assertIn("the CVV field", reason)

    def test_any_other_refusal_carries_its_own_observation(self):
        from yewo.agent.step.targeting import Decision, stalled_reason

        reason = stalled_reason(Decision(ok=False, observation="element 17 takes no text"))

        self.assertIn("element 17 takes no text", reason)

    def test_coordinates_still_work_when_marks_are_not_in_use(self):
        from yewo.agent.actuation.actions import Coordinates, TapAction
        from yewo.agent.step.targeting import evaluate

        action = TapAction(coordinates=Coordinates(x=5, y=6), reason="tap")

        self.assertTrue(evaluate(action, self._snapshot(), self._device()).ok)

    def test_an_untargeted_action_does_not_re_read_the_screen(self):
        """A key event is aimed at no point, so there is nothing to re-check."""
        from yewo.agent.actuation.actions import KeyEventAction
        from yewo.agent.step.targeting import evaluate

        device = self._device()

        decision = evaluate(KeyEventAction(key=4, reason="back"), self._snapshot(), device)

        self.assertTrue(decision.ok)
        device.get_screenshot.assert_not_called()

    def test_a_target_that_moved_stops_the_step(self):
        """The number resolved to a point, but the screen moved under it."""
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.step.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, target="Continue button", reason="continue"),
            self._snapshot(self._marks()),
            self._device(png=fakes.solid_png("black")),
        )

        self.assertFalse(decision.ok)
        self.assertIn("screen changed", decision.observation)

    def test_a_screen_that_cannot_be_read_again_stops_the_step(self):
        """No current screen is no evidence that the target survived."""
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.step.targeting import evaluate

        device = SimpleNamespace(
            get_screenshot=Mock(side_effect=ObservationUnavailable("session gone"))
        )

        decision = evaluate(
            TapMarkAction(mark=1, target="Continue button", reason="continue"),
            self._snapshot(self._marks()),
            device,
        )

        self.assertFalse(decision.ok)
        self.assertIn("could not be read", decision.observation)

    def test_an_empty_screenshot_stops_the_step(self):
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.step.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, target="Continue button", reason="continue"),
            self._snapshot(self._marks()),
            self._device(png=b""),
        )

        self.assertFalse(decision.ok)
        self.assertIn("could not be read", decision.observation)

    def test_resolving_a_number_leaves_the_proposed_action_untouched(self):
        """A proposed action is parsed once, never edited into a resolved one."""
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.step.targeting import evaluate

        proposed = TapMarkAction(mark=1, target="Continue button", reason="continue")

        decision = evaluate(proposed, self._snapshot(self._marks()), self._device())

        self.assertIsNot(decision.action, proposed)
        self.assertFalse(hasattr(proposed, "coordinates"))
        self.assertEqual(proposed.mark, 1)


class GroundingContractTests(unittest.TestCase):
    """Each mode accepts one way of naming a target, and refuses the other."""

    def _snapshot(self):
        return fakes.snapshot()

    def _device(self):
        return SimpleNamespace(get_screenshot=Mock(return_value=fakes.solid_png()))

    def _verdict(self, action, grounding):
        from yewo.agent.step.targeting import evaluate

        return evaluate(action, self._snapshot(), self._device(), grounding)

    def test_marks_mode_refuses_raw_coordinates(self):
        """Obeying them would reinstate the invented coordinate numbering prevents."""
        from yewo.agent.actuation.actions import Coordinates, TapAction
        from yewo.agent.plan.planner import GROUNDING_MARKS

        decision = self._verdict(
            TapAction(coordinates=Coordinates(x=535, y=1222), reason="r"),
            GROUNDING_MARKS,
        )

        self.assertFalse(decision.ok)
        self.assertIn("mark", decision.observation)

    def test_each_mode_makes_the_other_mode_unrepresentable(self):
        """Prose plus a runtime check is not enough — the model obeys the schema."""
        from pydantic import ValidationError

        from yewo.agent.actuation.actions import TapAction, TapMarkAction

        with self.assertRaises(ValidationError):
            TapAction(reason="r")  # coordinates are required
        with self.assertRaises(ValidationError):
            TapMarkAction(reason="r")  # a mark is required
        self.assertNotIn("coordinates", TapMarkAction.model_fields)
        self.assertNotIn("mark", TapAction.model_fields)

    def test_the_marks_schema_offers_no_coordinates_at_all(self):
        from yewo.agent.plan.planner import MarksStepResponse

        schema = MarksStepResponse.model_json_schema()
        tap = next(d for n, d in schema["$defs"].items() if n.startswith("Tap"))

        self.assertNotIn("coordinates", tap["properties"])
        self.assertIn("mark", tap["properties"])
        self.assertIn("mark", tap["required"])

    def test_an_untargeted_action_needs_neither(self):
        from yewo.agent.actuation.actions import KeyEventAction
        from yewo.agent.plan.planner import GROUNDING_MARKS

        self.assertTrue(self._verdict(KeyEventAction(key=4, reason="back"), GROUNDING_MARKS).ok)

    def test_the_heading_matches_the_mode(self):
        from yewo.agent.plan.planner import (
            GROUNDING_COORDINATES,
            GROUNDING_MARKS,
            DevicePlanner,
        )

        planner = DevicePlanner.__new__(DevicePlanner)
        planner.grounding = GROUNDING_COORDINATES
        self.assertNotIn("numbered", planner._target_heading())
        planner.grounding = GROUNDING_MARKS
        self.assertIn("numbered", planner._target_heading())

    def test_the_orchestrator_explains_why_not_listed_cannot_run(self):
        from yewo.agent.actuation.actions import TargetNotListedAction
        from yewo.agent.actuation.orchestrator import ActionOrchestrator

        orchestrator = ActionOrchestrator.__new__(ActionOrchestrator)
        orchestrator._scrubber = None

        with self.assertRaises(ValueError) as caught:
            orchestrator._run_action(TargetNotListedAction(description="blue button", reason="r"))

        self.assertIn("nothing to", str(caught.exception))

    def test_an_unmatched_screen_is_saved_redacted(self):
        from yewo.agent.artifacts.steps import save_unmatched_screen

        media = Mock()
        snapshot = SimpleNamespace(raw_img_bytes=b"png", gui_elements=["el"])

        save_unmatched_screen(media, Path("/tmp/x.png"), snapshot)

        media.save_redacted_bytes.assert_called_once_with(b"png", ["el"], Path("/tmp/x.png"))


class CoordinateShapeTests(unittest.TestCase):
    """The model answered with [x, y] and killed every scenario at step one."""

    def test_a_pair_is_read_as_coordinates(self):
        from yewo.agent.actuation.actions import Coordinates

        self.assertEqual(Coordinates.model_validate([537, 1260]).x, 537)
        self.assertEqual(Coordinates.model_validate((537, 1260)).y, 1260)

    def test_an_object_still_works(self):
        from yewo.agent.actuation.actions import Coordinates

        coordinates = Coordinates.model_validate({"x": 1, "y": 2})
        self.assertEqual((coordinates.x, coordinates.y), (1, 2))

    def test_a_whole_action_survives_the_pair_form(self):
        from pydantic import TypeAdapter

        from yewo.agent.actuation.actions import Action

        action = TypeAdapter(Action).validate_python(
            {"type": "tap", "coordinates": [537, 1260], "reason": "tap Login"}
        )

        self.assertEqual((action.coordinates.x, action.coordinates.y), (537, 1260))
