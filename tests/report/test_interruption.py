"""An interrupted scenario keeps the walk completed before cancellation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fakes import walk

from yewo.agent.memory.history import AgentHistory
from yewo.agent.run.outcomes import DeviceRunResult
from yewo.agent.run.record import StepRecord
from yewo.testing import interruption
from yewo.testing.models import GeneratedTestCase


def scenario() -> GeneratedTestCase:
    return GeneratedTestCase(
        test_id="TS_1",
        scenario_name="Create a schedule",
        cell="functional.happy_path.0",
        category="functional",
        scenario_type="happy_path",
        priority="P0",
        steps=walk("Open the form."),
        expected_outcomes=["The form opens."],
    )


class InterruptedScenarioTests(unittest.TestCase):
    def test_partial_walk_is_committed_before_remaining_scenarios_abort(self):
        manifest = Mock()
        usage = Mock()
        usage.take_calls_outside_steps.return_value = ["call"]
        run_result = DeviceRunResult(
            status="exception",
            steps_executed=2,
            failure_reason="interrupted",
            history=AgentHistory(),
            steps=[StepRecord(n=1), StepRecord(n=2)],
        )
        reports = []
        planned = scenario()

        interruption.record_active_scenario(
            manifest,
            planned,
            SimpleNamespace(
                _device_agent=SimpleNamespace(interrupted_result=run_result),
                _active_obligation_controller=None,
            ),
            KeyboardInterrupt("stopped"),
            reports,
            usage,
            ("planner",),
        )

        self.assertEqual(reports[0].steps_executed, 2)
        self.assertEqual([step.n for step in reports[0].steps], [1, 2])
        manifest.record.assert_called_once_with(
            planned.test_id, reports[0], calls=["call"]
        )

    def test_no_active_walk_consumes_no_unattached_calls(self):
        usage = Mock()

        interruption.record_active_scenario(
            Mock(),
            scenario(),
            SimpleNamespace(
                _device_agent=SimpleNamespace(interrupted_result=None),
                _active_obligation_controller=None,
            ),
            KeyboardInterrupt(),
            [],
            usage,
            ("planner",),
        )

        usage.take_calls_outside_steps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
