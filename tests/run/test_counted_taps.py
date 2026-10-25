"""Repetition as an instruction to the device, not a tally for the planner."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _tap(**fields):
    from yewo.agent.actuation.actions import Coordinates, TapAction

    return TapAction(coordinates=Coordinates(x=10, y=20), reason="r", **fields)


class TapCountTests(unittest.TestCase):
    """A planner asked to tap five times must judge how many have landed.

    The screen often does not show it, so the count becomes a guess that the
    next step has to re-make. Counted in one action, it is arithmetic.
    """

    def _taps(self, action):
        from yewo.agent.actuation.orchestrator import ActionOrchestrator

        device = SimpleNamespace(tap=Mock(), driver=Mock(), platform=None)
        with patch("yewo.agent.actuation.orchestrator.time.sleep"):
            ActionOrchestrator(device, scrubber=None).execute(action)
        return device.tap.call_count

    def test_one_tap_by_default(self):
        self.assertEqual(self._taps(_tap()), 1)

    def test_a_count_taps_that_many_times(self):
        self.assertEqual(self._taps(_tap(count=5)), 5)

    def test_the_count_is_bounded(self):
        """A scenario wanting hundreds of taps wants a different kind of test."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            _tap(count=1000)
        with self.assertRaises(ValidationError):
            _tap(count=0)

    def _interval_waits(self, action, seconds):
        """Sleeps of the interval length, ignoring the settle the loop always does."""
        from yewo.agent.actuation.orchestrator import ActionOrchestrator

        device = SimpleNamespace(tap=Mock(), driver=Mock(), platform=None)
        with patch("yewo.agent.actuation.orchestrator.time.sleep") as slept:
            ActionOrchestrator(device, scrubber=None).execute(action)
        return [call for call in slept.call_args_list if call.args == (seconds,)]

    def test_an_interval_waits_between_taps_and_not_before_the_first(self):
        waits = self._interval_waits(_tap(count=3, interval_ms=100), 0.1)

        self.assertEqual(len(waits), 2, "three taps have two gaps")

    def test_no_interval_means_no_waiting_between_taps(self):
        self.assertEqual(self._interval_waits(_tap(count=4), 0.0), [])

    def test_the_marks_variant_can_count_too(self):
        """Grounding changes how a target is named, not what the action does."""
        from yewo.agent.actuation.actions import TapMarkAction

        self.assertEqual(
            TapMarkAction(mark=3, target="Increment button", reason="r", count=5).count,
            5,
        )


class PlannerVocabularyTests(unittest.TestCase):
    def test_both_groundings_offer_the_count(self):
        from yewo.agent.plan.planner import GROUNDING_MARKS, DevicePlanner

        for grounding in (GROUNDING_MARKS, "coordinates"):
            planner = DevicePlanner.__new__(DevicePlanner)
            planner.grounding = grounding
            offered = planner._action_vocabulary() + planner._action_constraints()

            self.assertIn("`count`", offered, grounding)
            self.assertIn("Never count taps yourself", offered, grounding)
