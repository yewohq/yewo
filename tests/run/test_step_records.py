"""What each step of a run is written down as.

A report said a scenario failed and how many steps it took. It could not say
which step, what the agent believed at the time, what it aimed at, or what the
device did about it. These cases hold the record to carrying all four, to
carrying them when the run ends early, and to masking every declared secret
on the way.
"""

import unittest
from types import SimpleNamespace

from yewo.agent.actuation.actions import (
    ActionType,
    Coordinates,
    TapMarkAction,
    TapAction,
    TypeAction,
)
from yewo.agent.memory.history import ActionResult
from yewo.agent.run.record import CallRow, StepLog, StepRecord
from yewo.llm.usage import CallRecord
from yewo.utils.secrets import SecretScrubber


def planned(**extra):
    fields = {
        "current_screen": "Sign in",
        "evaluation_previous_goal": "Success — the app opened",
        "observation": "An email field and a password field",
        "reasoning": "the goal is to sign in",
        "goal_satisfied": False,
        "goal_evidence": "",
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def scrubbing() -> StepLog:
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"password": "password123"})
    return StepLog(scrubber.scrub)


class StepLogTests(unittest.TestCase):
    def test_nothing_is_recorded_before_a_step_is_open(self):
        """So a reader never sees a field that belongs to no step."""
        log = StepLog()
        log.observed("abc", 10.0)
        log.planned(planned(), 20.0)

        self.assertEqual(log.records, [])

    def test_each_step_is_numbered_from_one(self):
        log = StepLog()
        log.open(1)
        log.open(2)

        self.assertEqual([step.n for step in log.records], [1, 2])

    def test_a_step_that_was_only_perceived_keeps_the_rest_absent(self):
        """Absent means the step never got that far, and is not a result."""
        log = StepLog()
        log.open(1)
        log.observed("phash-1", 12.5)

        step = log.records[0]
        self.assertEqual(step.phash, "phash-1")
        self.assertEqual(step.perceive_ms, 12.5)
        self.assertIsNone(step.decision)
        self.assertIsNone(step.action)
        self.assertIsNone(step.outcome)
        self.assertIsNone(step.plan_ms)

    def test_all_five_prose_fields_are_kept(self):
        """The previous-step assessment is the one a single-field record drops."""
        log = StepLog()
        log.open(1)
        log.planned(planned(), 30.0)

        decision = log.records[0].decision
        self.assertEqual(decision.reasoning, "the goal is to sign in")
        self.assertEqual(
            decision.evaluation_previous_goal, "Success — the app opened"
        )
        self.assertEqual(decision.observation, "An email field and a password field")
        self.assertEqual(decision.current_screen, "Sign in")
        self.assertFalse(decision.goal_satisfied)
        self.assertEqual(log.records[0].screen_name, "Sign in")

    def test_every_prose_field_is_scrubbed(self):
        """All of them are persisted, and any of them can quote a typed value."""
        log = scrubbing()
        log.open(1)
        log.planned(
            planned(
                reasoning="I will type password123",
                observation="the field holds password123",
                evaluation_previous_goal="password123 was accepted",
                goal_evidence="password123 is masked",
            ),
            30.0,
        )

        written = log.records[0].decision.model_dump_json()
        self.assertNotIn("password123", written)
        self.assertIn("{{password}}", written)

    def test_an_action_that_reached_the_device_is_recorded_with_its_outcome(self):
        log = StepLog()
        log.open(1)
        log.record_action(
            TapAction(coordinates=Coordinates(x=10, y=20), mark=7, reason="place the order"),
            "tap (10, 20)",
            ActionResult(success=True, observation="tap executed — screen changed"),
            "Checkout",
        )

        step = log.records[0]
        self.assertEqual(step.action.type, ActionType.TAP.value)
        self.assertEqual(step.action.target_label, "Checkout")
        self.assertTrue(step.outcome.success)
        self.assertIsNone(step.outcome.error)

    def test_a_typed_value_does_not_reach_the_record(self):
        log = scrubbing()
        log.open(1)
        log.record_action(
            TypeAction(coordinates=Coordinates(x=1, y=2), text="password123", reason="sign in"),
            "type 'password123' at (1, 2)",
            ActionResult(success=False, observation="failed", error="password123 rejected"),
            None,
        )

        written = log.records[0].model_dump_json()
        self.assertNotIn("password123", written)

    def test_an_action_that_never_reached_the_device_is_still_a_step(self):
        """It happened to the step, and the grounding says why it stopped."""
        log = StepLog()
        log.open(1)
        log.record_grounding(
            "marks",
            SimpleNamespace(ok=False, observation="mark 9 is not listed", unmatched=True),
            TapMarkAction(mark=9, reason="open the menu"),
        )
        log.record_blocked_action(
            TapAction(coordinates=Coordinates(x=1, y=2), reason="open the menu"),
            "tap (1, 2)",
            "mark 9 is not listed",
        )

        step = log.records[0]
        self.assertFalse(step.grounding.resolved)
        self.assertTrue(step.grounding.unmatched)
        self.assertEqual(step.grounding.refusal, "mark 9 is not listed")
        self.assertFalse(step.outcome.success)

    def test_a_resolved_target_carries_no_refusal(self):
        log = StepLog()
        log.open(1)
        log.record_grounding(
            "marks",
            SimpleNamespace(ok=True, observation="matched", unmatched=False),
            TapMarkAction(mark=3, reason="tap Checkout"),
        )

        self.assertIsNone(log.records[0].grounding.refusal)
        self.assertEqual(log.records[0].grounding.mark_id, 3)

    def test_only_the_files_a_step_wrote_are_named(self):
        log = StepLog()
        log.open(1)
        log.files({"raw": "step001_raw.png", "marked": None})

        self.assertEqual(log.records[0].screenshots, {"raw": "step001_raw.png"})


class CallAttributionTests(unittest.TestCase):
    """Which step paid for which call."""

    def _call(self, **extra):
        fields = {
            "model": "openrouter/m",
            "served_model": "m",
            "generation_id": "gen-1",
            "attempt": 1,
            "finish_reason": "stop",
            "latency_ms": 120.0,
            "cost": 0.004,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        fields.update(extra)
        return CallRecord(**fields)

    def test_a_row_carries_the_accounting_and_not_the_answer(self):
        """The response body stays in the call log, so neither file repeats the other."""
        row = CallRow.of("planner", self._call(raw={"choices": ["a secret answer"]}))

        self.assertEqual(row.component, "planner")
        self.assertEqual(row.prompt_tokens, 100)
        self.assertNotIn("raw", row.model_dump())
        self.assertNotIn("secret answer", row.model_dump_json())

    def test_every_call_is_named_once(self):
        first = CallRow.of("planner", self._call())
        second = CallRow.of("planner", self._call())

        self.assertNotEqual(first.call_uuid, second.call_uuid)

    def test_a_call_lands_on_the_step_that_was_open(self):
        import tempfile
        from pathlib import Path

        from yewo.testing.report.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            first, second = StepRecord(n=1), StepRecord(n=2)

            tracker.mark(first)
            tracker.add_call("scenario:TS_1", self._call())
            tracker.mark(second)
            tracker.add_call("scenario:TS_1", self._call())
            tracker.add_call("scenario:TS_1", self._call())

        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 2)

    def test_a_call_made_between_steps_belongs_to_the_scenario(self):
        """The auditor and the route judge answer once the walk is over."""
        import tempfile
        from pathlib import Path

        from yewo.testing.report.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            step = StepRecord(n=1)
            tracker.mark(step)
            tracker.add_call("scenario:TS_1", self._call())
            tracker.mark(None)
            tracker.add_call("scenario:TS_1", self._call())

            outside = tracker.calls_outside_steps("scenario:TS_1")

        self.assertEqual(len(step.calls), 1)
        self.assertEqual(len(outside), 1)
        self.assertEqual(tracker.totals.total_calls, 2)

    def test_the_calls_recorded_add_up_to_what_the_run_spent(self):
        """Otherwise a reader has to know which judges answer after the loop."""
        import tempfile
        from pathlib import Path

        from yewo.testing.report.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            step = StepRecord(n=1)
            tracker.mark(step)
            for _ in range(3):
                tracker.add_call("scenario:TS_1", self._call())
            tracker.mark(None)
            tracker.add_call("scenario:TS_1", self._call())

            counted = len(step.calls) + len(
                tracker.calls_outside_steps("scenario:TS_1")
            )

        self.assertEqual(counted, tracker.totals.total_calls)


class ScreenNameTests(unittest.TestCase):
    """A screen name is prose, and prose carries whatever is on the screen.

    A model names a screen from what it can see. A screen that greets an
    account by name, or puts a value in its title, puts that value in the
    name it is given.
    """

    def test_the_name_a_model_gave_the_screen_is_scrubbed(self):
        log = scrubbing()
        log.open(1)
        log.planned(planned(current_screen="Signed in as password123"), 1.0)

        step = log.records[0]
        self.assertNotIn("password123", step.screen_name)
        self.assertNotIn("password123", step.decision.current_screen)
        self.assertIn("{{password}}", step.screen_name)

    def test_the_two_places_it_is_written_agree(self):
        log = scrubbing()
        log.open(1)
        log.planned(planned(current_screen="Signed in as password123"), 1.0)

        self.assertEqual(
            log.records[0].screen_name, log.records[0].decision.current_screen
        )


class TargetLabelTests(unittest.TestCase):
    """What an element said is app content too."""

    def test_the_label_of_a_target_that_was_acted_on_is_scrubbed(self):
        from yewo.agent.memory.history import ActionResult

        log = scrubbing()
        log.open(1)
        log.record_action(
            TapAction(coordinates=Coordinates(x=1, y=2), reason="pay"),
            "tap",
            ActionResult(success=True, observation="ok"),
            "Confirm password123",
        )

        self.assertNotIn("password123", log.records[0].action.target_label)

    def test_a_refused_action_still_records_what_it_aimed_at(self):
        log = StepLog()
        log.open(1)
        log.record_blocked_action(
            TapAction(coordinates=Coordinates(x=1, y=2), reason="pay"),
            "tap",
            "mark 9 is not listed",
            "Place Order",
        )

        step = log.records[0]
        self.assertEqual(step.action.target_label, "Place Order")
        self.assertEqual(step.outcome.error, "mark 9 is not listed")
        self.assertFalse(step.outcome.success)

    def test_an_element_with_no_label_records_none_rather_than_empty(self):
        from yewo.agent.memory.history import ActionResult

        log = StepLog()
        log.open(1)
        log.record_action(
            TapAction(coordinates=Coordinates(x=1, y=2), reason="pay"),
            "tap",
            ActionResult(success=True, observation="ok"),
            "",
        )

        self.assertIsNone(log.records[0].action.target_label)


class ComponentTests(unittest.TestCase):
    """Which component made a call, when two of them share one client."""

    def test_the_shape_asked_for_tells_the_planner_from_the_auditor(self):
        from yewo.llm.usage import CallRecord

        def row(answering):
            return CallRow.of(
                "agent",
                CallRecord(
                    model="m", served_model="m", generation_id="g", attempt=1,
                    finish_reason="stop", latency_ms=1.0, cost=0.0,
                    answering=answering,
                ),
            )

        self.assertEqual(row("QAStepResponse").answering, "QAStepResponse")
        self.assertNotEqual(
            row("QAStepResponse").answering,
            row("ScreenAssessment").answering,
        )

    def test_a_call_with_no_required_shape_names_none(self):
        from yewo.llm.usage import CallRecord

        row = CallRow.of(
            "agent",
            CallRecord(
                model="m", served_model="m", generation_id="g", attempt=1,
                finish_reason="stop", latency_ms=1.0, cost=0.0,
            ),
        )

        self.assertIsNone(row.answering)
