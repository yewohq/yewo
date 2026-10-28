"""Compatibility checks for the Yewo AndroidWorld agent adapter."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "android_world"))

from yewo_agent import YewoAndroidWorldAgent


class YewoAndroidWorldAgentTests(unittest.TestCase):
    def test_run_adb_uses_async_environment_controller(self):
        controller = object()
        agent = object.__new__(YewoAndroidWorldAgent)
        agent._env = SimpleNamespace(controller=controller)
        response = SimpleNamespace(generic=SimpleNamespace(output=b"123\n"))

        with mock.patch("yewo_agent.adb_utils.issue_generic_request", return_value=response) as issue:
            output = agent._run_adb(["shell", "pidof", "package"])

        issue.assert_called_once_with(["shell", "pidof", "package"], controller)
        self.assertEqual(output, "123\n")

    def setUp(self) -> None:
        rebind = mock.patch("yewo_agent.ensure_forwarder_bound")
        self.rebind = rebind.start()
        self.addCleanup(rebind.stop)

    def _agent(self, run=None) -> YewoAndroidWorldAgent:
        agent = object.__new__(YewoAndroidWorldAgent)
        agent._env = SimpleNamespace(
            hide_automation_ui=mock.Mock(), controller=object()
        )
        agent._goal = "goal"
        agent._completed_run = None
        agent._step_budget = None
        agent._task = SimpleNamespace(
            step=mock.Mock(return_value=run),
            context=SimpleNamespace(steps_executed=1),
        )
        agent._usage = SimpleNamespace(
            checkpoint=mock.Mock(), summary=mock.Mock(return_value={})
        )
        agent._run_adb = mock.Mock()
        return agent

    def test_each_step_hides_android_world_automation_ui(self):
        agent = self._agent()

        result = agent.step("goal")

        agent.env.hide_automation_ui.assert_called_once_with()
        self.assertFalse(result.done)

    def test_a_reached_goal_is_reported_as_done(self):
        agent = self._agent(run=_run("goal_reached"))

        self.assertTrue(agent.step("goal").done)

    def test_an_exhausted_budget_is_reported_as_done_for_evaluation(self):
        """AndroidWorld scores the task state only when interaction_results.done is True.

        When the episode's step budget is exhausted, the agent has finished all
        planned actions. Reporting done=True allows AndroidWorld to evaluate
        task.is_successful(env) against ground truth instead of unconditionally
        penalizing the trial with 0.0.
        """
        agent = self._agent(run=_run("step_limit", "Reached maximum step limit"))

        result = agent.step("goal")

        self.assertTrue(result.done)
        self.assertEqual(result.data["yewo_status"], "step_limit")

    def test_a_detected_loop_is_not_reported_as_done(self):
        agent = self._agent(run=_run("loop_detected", "Target was not detected"))

        self.assertFalse(agent.step("goal").done)

    def test_the_forwarder_is_rebound_once_for_a_finished_task(self):
        """The rebind prepares AndroidWorld's evaluator, and costs an adb call.

        A task that ends before the episode does is stepped again for every
        remaining turn, so repeating the rebind would spend an adb round trip
        on each of them.
        """
        agent = self._agent(run=_run("step_limit"))

        agent.step("goal")
        agent.step("goal")

        self.rebind.assert_called_once_with(agent._run_adb)

    def test_android_worlds_budget_becomes_yewos_step_limit(self):
        """Yewo's own ceiling would otherwise end a task the budget allowed."""
        agent = self._agent()
        agent._agent = SimpleNamespace(start_task=mock.Mock())
        agent._device_control = object()
        agent._perception = object()
        agent._session = SimpleNamespace(screenshots_dir=Path("/tmp/screens"))
        agent._task = None

        agent.set_max_steps(22)
        agent.step("a new goal")

        self.assertEqual(
            agent._agent.start_task.call_args.kwargs["max_steps"], 22
        )


def _run(status: str, failure_reason: str | None = None) -> SimpleNamespace:
    """What DeviceTask.step returns once the task has ended."""
    return SimpleNamespace(status=status, failure_reason=failure_reason)


if __name__ == "__main__":
    unittest.main()
