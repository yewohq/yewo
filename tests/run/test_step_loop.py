"""One real turn of the loop, and what it leaves written down.

The step log on its own proves only that the log agrees with itself. These
drive the agent with fakes at the device, the perception and the planner, so
a stage that stops filling the record fails here.
"""

import unittest
from types import SimpleNamespace

from yewo.utils.secrets import SecretScrubber


class LoopRecordTests(unittest.TestCase):
    """One step, driven through `DeviceAgent._run_step`."""

    def _snapshot(self):
        import io

        from PIL import Image

        from yewo.agent.step.observation import PerceptionSnapshot

        buffer = io.BytesIO()
        image = Image.new("RGB", (4, 4))
        image.save(buffer, format="PNG")
        return PerceptionSnapshot(
            gui_elements=[],
            marks=[],
            raw_img_bytes=buffer.getvalue(),
            planner_img_bytes=buffer.getvalue(),
            annotated_img=image,
            current_phash="phash-1",
            timestamp="20261021_120000",
        )

    def _agent(self, response):
        from unittest.mock import Mock

        from yewo.agent.grounding import GROUNDING_COORDINATES
        from yewo.agent.run.agent import DeviceAgent

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.scrubber = SecretScrubber()
        agent.scrubber.learn_secrets({"password": "password123"})
        agent.grounding = GROUNDING_COORDINATES
        agent.grounder = None
        agent.settle_time = 0.0
        agent.replay_ledger = None
        agent.max_steps = 10
        agent.stop_when_satisfied = False
        agent.artifacts = Mock()
        agent.images = Mock()
        agent.artifacts.save_step.return_value = {"raw": "step001_raw.png"}
        agent.artifacts.save_planner_view.return_value = None
        agent.artifacts.save_action.return_value = "step001_action.png"
        agent._default_fast_planner = Mock()
        agent._default_fast_planner.plan_step.return_value = response
        return agent

    def _response(self, action):
        from yewo.agent.plan.schema import DeviceStepResponse

        return DeviceStepResponse(
            current_screen="Sign in",
            evaluation_previous_goal="Success — the app opened",
            observation="An email field and a password field",
            reasoning="type password123 into the field",
            action=action,
        )

    def _run_one(
        self,
        action,
        observation_hook=None,
        pending_entered_value=None,
        focused_field=None,
    ):
        from pathlib import Path
        from unittest.mock import Mock, patch

        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.memory.loop_detector import RollingLoopDetector
        from yewo.agent.run import record as record_module
        from yewo.agent.run.context import RunContext

        agent = self._agent(self._response(action))
        ctx = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=record_module.StepLog(agent.scrubber.scrub),
        )
        ctx.pending_entered_value = pending_entered_value
        device = Mock()
        device.is_keyboard_open.return_value = False
        device.read_focused_field.return_value = focused_field
        agent.durable_replay = None

        with (
            patch("yewo.agent.step.observation.capture", return_value=self._snapshot()),
            patch("yewo.agent.step.dispatch.settle.settle", return_value="phash-2"),
            patch("yewo.agent.step.dispatch.ActionOrchestrator") as orchestrator,
            patch("yewo.agent.device.state.context_drift", return_value=None),
        ):
            orchestrator.return_value.execute.return_value = None
            agent._run_step(
                step=0,
                goal="sign in",
                planner=agent._default_fast_planner,
                device_control=device,
                perception=Mock(),
                session_dir=Path("."),
                session_trail=None,
                observation_hook=observation_hook,
                step_hook=None,
                ctx=ctx,
            )
        return ctx.steps.records

    def test_one_step_leaves_one_complete_record(self):
        from yewo.agent.actuation.actions import Coordinates
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction

        records = self._run_one(
            TapCoordinateAction(
                coordinates=Coordinates(x=10, y=20),
                target="Sign in button",
                reason="sign in",
            )
        )

        self.assertEqual(len(records), 1)
        step = records[0]
        self.assertEqual(step.n, 1)
        self.assertEqual(step.phash, "phash-1")
        self.assertEqual(step.screen_name, "Sign in")
        self.assertIsNotNone(step.decision)
        self.assertIsNotNone(step.action)
        self.assertIsNotNone(step.grounding)
        self.assertIsNotNone(step.outcome)
        self.assertTrue(step.outcome.success)
        self.assertEqual(step.screenshots["raw"], "step001_raw.png")
        self.assertEqual(step.screenshots["action"], "step001_action.png")

    def test_the_time_each_stage_took_is_recorded(self):
        from yewo.agent.actuation.actions import Coordinates
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction

        step = self._run_one(
            TapCoordinateAction(
                coordinates=Coordinates(x=10, y=20),
                target="Sign in button",
                reason="sign in",
            )
        )[0]

        for name in ("perceive_ms", "plan_ms", "act_ms"):
            self.assertIsNotNone(getattr(step, name), name)

    def test_a_declared_secret_in_the_reasoning_never_reaches_the_record(self):
        from yewo.agent.actuation.actions import Coordinates
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction

        step = self._run_one(
            TapCoordinateAction(
                coordinates=Coordinates(x=10, y=20),
                target="Sign in button",
                reason="sign in",
            )
        )[0]

        self.assertNotIn("password123", step.model_dump_json())
        self.assertIn("{{password}}", step.decision.reasoning)

    def test_an_unlisted_target_is_recorded_as_a_step_that_reached_nothing(self):
        from yewo.agent.actuation.actions import TargetNotListedAction

        step = self._run_one(TargetNotListedAction(description="a Pay button", reason="pay"))[0]

        self.assertFalse(step.grounding.resolved)
        self.assertTrue(step.grounding.unmatched)
        self.assertFalse(step.outcome.success)
        self.assertIsNotNone(step.decision)

    def test_an_observation_stop_prevents_cache_selection_and_planning(self):
        from unittest.mock import patch

        from yewo.agent.actuation.actions import Coordinates
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
        from yewo.agent.run.outcomes import ObservationDirective

        action = TapCoordinateAction(
            coordinates=Coordinates(x=10, y=20),
            target="Sign in button",
            reason="sign in",
        )
        with patch("yewo.agent.run.saved_path.try_cached_step") as cached:
            records = self._run_one(
                action,
                observation_hook=lambda _snapshot, _step: ObservationDirective.STOP,
            )

        cached.assert_not_called()
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].decision)
        self.assertIsNone(records[0].action)

    def test_a_failed_entry_forces_planner_recovery_before_more_replay(self):
        from unittest.mock import patch

        from yewo.agent.actuation.actions import Coordinates, TypeAction
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
        from yewo.agent.device.focus import FocusedFieldReading
        from yewo.agent.memory.replay_store import ValueNames
        from yewo.agent.run.postconditions import pending_for

        values = {"amount": "125.00"}
        pending = pending_for(
            TypeAction(
                text="125.00",
                coordinates=Coordinates(x=10, y=20),
                target="Amount",
                reason="Enter the amount.",
            ),
            ValueNames(values),
            values,
            "12.50",
        )
        recovery = TapCoordinateAction(
            coordinates=Coordinates(x=10, y=20),
            target="Amount",
            reason="Recover the entry.",
        )
        with patch("yewo.agent.run.saved_path.try_cached_step", return_value=True) as cached:
            records = self._run_one(
                recovery,
                pending_entered_value=pending,
                focused_field=FocusedFieldReading("12.50", (1, 1, 20, 30)),
            )

        cached.assert_not_called()
        self.assertIsNotNone(records[0].decision)


class MarkerLifetimeTests(unittest.TestCase):
    """Who a call belongs to once the loop is over.

    The auditor and the route judge both answer after the last step. Charged
    to that step, the record would say a step paid for work done after it.
    """

    def test_the_marker_is_cleared_when_the_run_ends(self):
        from unittest.mock import Mock, patch

        from yewo.agent.run.agent import DeviceAgent

        marked = []
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.scrubber = SecretScrubber()
        agent.replay_ledger = None
        agent.max_steps = 1
        agent._default_fast_planner = Mock()
        agent.step_opened = marked.append

        with (
            patch("yewo.agent.run.agent.restart_app"),
            patch(
                "yewo.agent.run.agent.replay_known_path",
                side_effect=RuntimeError("stop here"),
            ),
        ):
            agent.run(
                goal="g",
                device_control=SimpleNamespace(screen_height=2400),
                perception=Mock(),
                session_dir=Mock(),
            )

        self.assertEqual(marked[-1], None)

    def test_the_marker_is_cleared_even_when_the_run_raises(self):
        from unittest.mock import Mock, patch

        from yewo.agent.run.agent import DeviceAgent
        from yewo.errors import ProviderUnavailable

        marked = []
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.scrubber = SecretScrubber()
        agent.replay_ledger = None
        agent.max_steps = 1
        agent._default_fast_planner = Mock()
        agent.step_opened = marked.append

        with (
            patch("yewo.agent.run.agent.restart_app"),
            patch(
                "yewo.agent.run.agent.replay_known_path",
                side_effect=ProviderUnavailable("spent"),
            ),
            self.assertRaises(ProviderUnavailable),
        ):
            agent.run(
                goal="g",
                device_control=SimpleNamespace(screen_height=2400),
                perception=Mock(),
                session_dir=Mock(),
            )

        self.assertEqual(marked[-1], None)


class FinalObservationTests(unittest.TestCase):
    def test_a_final_audit_provider_error_is_not_hidden_as_missing_learning(self):
        from unittest.mock import Mock, patch

        from fakes import snapshot
        from yewo.agent.actuation.actions import Coordinates, TypeAction
        from yewo.agent.device.focus import FocusedFieldReading
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.memory.loop_detector import RollingLoopDetector
        from yewo.agent.memory.replay_store import ValueNames
        from yewo.agent.run import record as record_module, saved_path
        from yewo.agent.run.context import RunContext
        from yewo.agent.run.postconditions import pending_for
        from yewo.errors import ProviderUnavailable

        values = {"reference": "AB123"}
        pending = pending_for(
            TypeAction(
                text="AB123",
                coordinates=Coordinates(x=10, y=20),
                target="Reference",
                reason="Enter the reference.",
            ),
            ValueNames(values),
            values,
            "AB123",
        )
        ctx = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=record_module.StepLog(),
            pending_entered_value=pending,
        )
        device = Mock()
        device.read_focused_field.return_value = FocusedFieldReading(
            "AB123", (1, 1, 20, 30)
        )
        agent = SimpleNamespace(
            durable_replay=None,
            scrubber=SecretScrubber(),
        )

        with (
            patch(
                "yewo.agent.run.saved_path.observation.capture",
                return_value=snapshot(),
            ),
            self.assertRaises(ProviderUnavailable),
        ):
            saved_path.record_arrival(
                agent,
                device,
                Mock(),
                None,
                ctx,
                lambda _screen, _step: (_ for _ in ()).throw(
                    ProviderUnavailable("audit unavailable")
                ),
            )

class FailedActionTests(unittest.TestCase):
    """An action that raises is still an action that happened.

    The error is kept, the screen is settled anyway, and the stall policy is
    told. A step that reached nothing stalls a screen as surely as one refused
    before dispatch, and the record has to say which it was.
    """

    def _dispatched(self, failure):
        from unittest.mock import Mock, patch

        from yewo.agent.actuation.actions import Coordinates
        from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.memory.loop_detector import RollingLoopDetector
        from yewo.agent.plan.schema import DeviceStepResponse
        from yewo.agent.run.context import RunContext
        from yewo.agent.run.record import StepLog
        from yewo.agent.step import dispatch

        action = TapCoordinateAction(
            coordinates=Coordinates(x=1, y=2),
            target="Sign in button",
            reason="sign in",
        )
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "password123"})
        ctx = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=StepLog(scrubber.scrub),
        )
        ctx.steps.open(1)

        with (
            patch("yewo.agent.step.dispatch.settle.settle", return_value="phash-2"),
            patch("yewo.agent.step.dispatch.ActionOrchestrator") as orchestrator,
        ):
            orchestrator.return_value.execute.side_effect = failure
            dispatch.execute(
                step=0,
                action=action,
                step_response=DeviceStepResponse(
                    current_screen="Sign in",
                    observation="a form",
                    reasoning="sign in",
                    action=action,
                ),
                device_control=Mock(),
                snapshot=SimpleNamespace(current_phash="phash-1", gui_elements=[], marks=[]),
                session_trail=None,
                ctx=ctx,
                scrubber=scrubber,
                replay_ledger=None,
                settle_time=0.0,
            )
        return ctx

    def test_an_action_that_raised_is_recorded_as_a_failure(self):
        ctx = self._dispatched(RuntimeError("the driver dropped the session"))
        step = ctx.steps.records[0]

        self.assertFalse(step.outcome.success)
        self.assertIn("dropped the session", step.outcome.error)

    def test_the_screen_is_marked_as_going_nowhere(self):
        ctx = self._dispatched(RuntimeError("no"))

        self.assertTrue(ctx.stall.is_blocked)
        self.assertEqual(ctx.stall.count, 1)

    def test_a_target_that_took_nothing_is_recorded_the_same_way(self):
        from yewo.errors import TargetNotInteractive

        ctx = self._dispatched(TargetNotInteractive("the field took no text"))
        step = ctx.steps.records[0]

        self.assertFalse(step.outcome.success)
        self.assertIn("took no text", step.outcome.error)

    def test_a_declared_value_in_a_device_error_is_masked(self):
        ctx = self._dispatched(RuntimeError("could not type password123"))

        self.assertNotIn("password123", ctx.steps.records[0].model_dump_json())

    def test_the_planner_is_told_the_action_failed(self):
        """It reads the history next, and a silent failure reads as a success."""
        ctx = self._dispatched(RuntimeError("the driver dropped the session"))

        self.assertIn("error", ctx.history.format_for_prompt().lower())
