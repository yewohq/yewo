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

    def _run_one(self, action):
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
        device = Mock()
        device.is_keyboard_open.return_value = False

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
                step_hook=None,
                ctx=ctx,
            )
        return ctx.steps.records

    def test_one_step_leaves_one_complete_record(self):
        from yewo.agent.actuation.actions import Coordinates, TapAction

        records = self._run_one(
            TapAction(coordinates=Coordinates(x=10, y=20), reason="sign in")
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
        from yewo.agent.actuation.actions import Coordinates, TapAction

        step = self._run_one(
            TapAction(coordinates=Coordinates(x=10, y=20), reason="sign in")
        )[0]

        for name in ("perceive_ms", "plan_ms", "act_ms"):
            self.assertIsNotNone(getattr(step, name), name)

    def test_a_declared_secret_in_the_reasoning_never_reaches_the_record(self):
        from yewo.agent.actuation.actions import Coordinates, TapAction

        step = self._run_one(
            TapAction(coordinates=Coordinates(x=10, y=20), reason="sign in")
        )[0]

        self.assertNotIn("password123", step.model_dump_json())
        self.assertIn("{{password}}", step.decision.reasoning)

    def test_an_unlisted_target_is_recorded_as_a_step_that_reached_nothing(self):
        from yewo.agent.actuation.actions import TargetNotListedAction

        step = self._run_one(
            TargetNotListedAction(description="a Pay button", reason="pay")
        )[0]

        self.assertFalse(step.grounding.resolved)
        self.assertTrue(step.grounding.unmatched)
        self.assertFalse(step.outcome.success)
        self.assertIsNotNone(step.decision)


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


class FailedActionTests(unittest.TestCase):
    """An action that raises is still an action that happened.

    The error is kept, the screen is settled anyway, and the stall policy is
    told. A step that reached nothing stalls a screen as surely as one refused
    before dispatch, and the record has to say which it was.
    """

    def _dispatched(self, failure):
        from unittest.mock import Mock, patch

        from yewo.agent.actuation.actions import Coordinates, TapAction
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.memory.loop_detector import RollingLoopDetector
        from yewo.agent.plan.schema import DeviceStepResponse
        from yewo.agent.run.context import RunContext
        from yewo.agent.run.record import StepLog
        from yewo.agent.step import dispatch

        action = TapAction(coordinates=Coordinates(x=1, y=2), reason="sign in")
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
                snapshot=SimpleNamespace(
                    current_phash="phash-1", gui_elements=[], marks=[]
                ),
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
