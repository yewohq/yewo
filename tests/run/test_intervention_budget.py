"""Repair spends its own allowance, never the steps an instruction asked for."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.device.launch import SetupMode
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.run.agent import DeviceAgent
from yewo.agent.run.context import RunContext
from yewo.agent.run.control_discovery import (
    ControlDiscoveryState,
    ControlDiscoveryTransaction,
)
from yewo.agent.run.recovery import RecoveryCause, RecoveryTransaction
from yewo.utils.secrets import SecretScrubber


def _context() -> RunContext:
    return RunContext(history=AgentHistory(), loop_detector=RollingLoopDetector())


def _probing() -> ControlDiscoveryTransaction:
    return ControlDiscoveryTransaction(
        source_phash="0" * 64,
        region="lower",
        description="the add button",
        intent="Add one exercise.",
    )


class ChargingTests(unittest.TestCase):
    def test_an_ordinary_turn_is_charged_to_the_instruction(self):
        ctx = _context()

        ctx.spend_step()

        self.assertEqual(ctx.steps_executed, 1)
        self.assertEqual(ctx.intervention_steps, 0)
        self.assertEqual(ctx.turns_spent, 1)

    def test_a_probe_turn_is_charged_to_repair(self):
        ctx = _context()
        ctx.control_discovery = _probing()

        ctx.spend_step()

        self.assertEqual(ctx.steps_executed, 0)
        self.assertEqual(ctx.intervention_steps, 1)

    def test_a_settled_probe_returns_the_turn_to_the_instruction(self):
        ctx = _context()
        probe = _probing()
        probe.state = ControlDiscoveryState.RESOLVED
        ctx.control_discovery = probe

        ctx.spend_step()

        self.assertEqual(ctx.steps_executed, 1)
        self.assertEqual(ctx.intervention_steps, 0)

    def test_a_recovery_turn_is_charged_to_repair(self):
        ctx = _context()
        ctx.active_recovery = RecoveryTransaction.open(
            failed_step=1,
            opened_step=2,
            original_intent="Enter the name.",
            failure_reason="The original action did not satisfy its postcondition.",
            cause=RecoveryCause.SEMANTIC_OBSTRUCTION,
        )

        ctx.spend_step()

        self.assertEqual(ctx.steps_executed, 0)
        self.assertEqual(ctx.intervention_steps, 1)


class LoopBoundTests(unittest.TestCase):
    """The walk ends on the budget whose work it actually ran out of."""

    def _agent(self, on_step):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.max_steps = 2
        agent.settle_time = 0
        agent.replay_ledger = None
        agent._default_fast_planner = Mock()
        agent.scrubber = SecretScrubber()
        agent.step_opened = Mock()
        agent.declared_values = {}
        agent.value_verifications = {}
        agent.value_names = ValueNames()
        agent._run_step = Mock(side_effect=on_step)
        return agent

    def _run_to_terminal(self, on_step, max_steps=2):
        agent = self._agent(on_step)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.run.task.replay_known_path"),
            patch("yewo.agent.run.task.saved_path.record_arrival", return_value=False),
        ):
            task = agent.start_task(
                goal="Add one exercise",
                device_control=SimpleNamespace(screen_height=2400),
                perception=Mock(),
                session_dir=Path(directory),
                max_steps=max_steps,
                setup=SetupMode.PRESERVE,
                replay=False,
            )
            for _ in range(200):
                result = task.step()
                if result is not None:
                    return result, task
        raise AssertionError("The walk never reached a terminal.")

    def test_probing_does_not_end_the_walk_at_its_step_limit(self):
        def probe(**kwargs):
            context = kwargs["ctx"]
            context.control_discovery = _probing()
            context.spend_step()
            return None

        result, task = self._run_to_terminal(probe)

        self.assertEqual(result.status, "intervention_limit")
        self.assertEqual(result.steps_executed, 0)
        self.assertEqual(result.intervention_steps, task.intervention_limit)
        self.assertIn("without resuming the instruction", result.failure_reason)

    def test_instruction_steps_still_end_the_walk_at_the_step_limit(self):
        def act(**kwargs):
            kwargs["ctx"].spend_step()
            return None

        result, _ = self._run_to_terminal(act)

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps_executed, 2)
        self.assertEqual(result.intervention_steps, 0)

    def test_a_turn_charging_nothing_is_still_the_instruction_s_turn(self):
        result, _ = self._run_to_terminal(lambda **_: None)

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps_executed, 2)
