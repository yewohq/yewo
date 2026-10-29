"""A benchmark advances Yewo by one fair turn at a time."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.device.launch import SetupMode
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.run.agent import DeviceAgent
from yewo.agent.run.outcomes import DeviceRunResult
from yewo.utils.secrets import SecretScrubber


def completed_result(context):
    return DeviceRunResult(
        status="goal_reached",
        steps_executed=context.steps_executed,
        failure_reason=None,
        history=context.history,
    )


class ResumableTaskTests(unittest.TestCase):
    def _agent(self, side_effect):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.max_steps = 3
        agent.settle_time = 0
        agent.replay_ledger = None
        agent._default_fast_planner = Mock()
        agent.scrubber = SecretScrubber()
        agent.step_opened = Mock()
        agent.declared_values = {}
        agent.value_verifications = {}
        agent.value_names = ValueNames()
        agent._run_step = Mock(side_effect=side_effect)
        return agent

    def _task(self, agent, directory, max_steps=3, replay=True):
        return agent.start_task(
            goal="Create a contact for Ada",
            device_control=SimpleNamespace(screen_height=2400),
            perception=Mock(),
            session_dir=Path(directory),
            max_steps=max_steps,
            setup=SetupMode.PRESERVE,
            replay=replay,
        )

    def test_replay_can_be_disabled_for_benchmark_turn_accounting(self):
        agent = self._agent(None)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path") as replay,
        ):
            self._task(agent, directory, replay=False)

        replay.assert_not_called()

    def test_task_owns_instruction_and_execution_context_identities(self):
        agent = self._agent(None)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
        ):
            task = self._task(agent, directory, replay=False)

        controller = task.context.instruction_controller
        execution_context = task.context.execution_context
        self.assertIsNotNone(controller)
        self.assertTrue(controller.shadow_mode)
        self.assertEqual(controller.active_index, 1)
        self.assertIsNotNone(execution_context)
        self.assertEqual(execution_context.setup, SetupMode.PRESERVE.value)
        self.assertTrue(execution_context.context_id)

    def test_each_call_advances_at_most_one_agent_turn(self):
        def run_step(**kwargs):
            if kwargs["step"] == 1:
                return completed_result(kwargs["ctx"])
            return None

        agent = self._agent(run_step)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
            patch("yewo.agent.run.task.saved_path.record_arrival", return_value=False),
        ):
            task = self._task(agent, directory)

            self.assertIsNone(task.step())
            self.assertEqual(agent._run_step.call_count, 1)
            result = task.step()

        self.assertEqual(result.status, "goal_reached")
        self.assertEqual(agent._run_step.call_count, 2)

    def test_interruption_preserves_the_active_step(self):
        def interrupt(**kwargs):
            context = kwargs["ctx"]
            context.steps_executed += 1
            context.steps.open(1)
            raise KeyboardInterrupt("stop")

        agent = self._agent(interrupt)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
            self.assertRaises(KeyboardInterrupt),
        ):
            agent.run(
                goal="Create a contact for Ada",
                device_control=SimpleNamespace(screen_height=2400),
                perception=Mock(),
                session_dir=Path(directory),
                max_steps=3,
                setup=SetupMode.PRESERVE,
            )

        self.assertIsNotNone(agent.interrupted_result)
        self.assertEqual(agent.interrupted_result.steps_executed, 1)
        self.assertEqual(len(agent.interrupted_result.steps), 1)

    def test_step_limit_is_owned_by_the_resumable_task(self):
        agent = self._agent([None, None])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
            patch("yewo.agent.run.task.saved_path.record_arrival", return_value=False),
        ):
            task = self._task(agent, directory, max_steps=2)
            self.assertIsNone(task.step())
            result = task.step()

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(agent._run_step.call_count, 2)

    def test_finished_task_does_not_act_again(self):
        def finish(**kwargs):
            return completed_result(kwargs["ctx"])

        agent = self._agent(finish)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
            patch("yewo.agent.run.task.saved_path.record_arrival", return_value=False),
        ):
            task = self._task(agent, directory)
            first = task.step()
            second = task.step()

        self.assertIs(first, second)
        agent._run_step.assert_called_once()
