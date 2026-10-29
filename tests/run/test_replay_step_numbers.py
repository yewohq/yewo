"""Which step an issue is stamped with, on a run that replayed part of it.

A report matches an issue to a step. The step a cached run walked never
reaches `qa_outputs`, because no planner filled it, so a count of planned
steps alone names the wrong one. An issue found on the third screen of a
run that cached two steps would be reported as the first.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.memory.history import AgentHistory
from yewo.agent.plan.schema import DoneAction
from yewo.agent.memory.replay_learning import DurableReplay
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.plan.planner import GROUNDING_COORDINATES
from yewo.agent.run.agent import DeviceRunResult
from yewo.agent.device.platform import Platform
from yewo.testing.evidence import QAIssue
from yewo.testing.judge.qa_auditor import ScreenAssessment
from yewo.testing.models import GeneratedTestCase
from yewo.testing.qa_agent import QAAgent
from yewo.testing.qa_planner import QAStepResponse
from yewo.utils.secrets import SecretScrubber

from fakes import walk


def replay_of(cached_steps):
    replay = DurableReplay(trace=None, values={}, names=ValueNames())
    replay.cached_step_count = cached_steps
    return replay


class StampedIssueTests(unittest.TestCase):
    """The count, as the scenario actually uses it."""

    def _agent(self, replay):
        agent = QAAgent.__new__(QAAgent)
        agent._fast_llm = object()
        agent._grounding = GROUNDING_COORDINATES
        agent._trace_store = None
        agent._build_replay = lambda *args, **kwargs: replay
        agent._device_agent = SimpleNamespace(
            max_steps=40,
            replay_ledger=None,
            durable_replay=None,
            scrubber=SecretScrubber(),
            run=Mock(side_effect=self._one_planned_step),
        )
        return agent

    @staticmethod
    def _one_planned_step(**kwargs):
        snapshot = SimpleNamespace(current_phash="p", raw_img_bytes=b"", scan_info=None)
        kwargs["observation_hook"](snapshot, 3)
        kwargs["step_hook"](
            QAStepResponse(
                current_screen="Checkout",
                observation="what is on it",
                reasoning="why this tap",
                action=DoneAction(reason="the order is placed"),
                is_scenario_feasible=True,
            ),
            snapshot,
        )
        return DeviceRunResult(
            status="step_limit",
            steps_executed=1,
            failure_reason="limit",
            history=AgentHistory(),
        )

    def _case(self):
        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Checkout",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Tap Pay."),
            expected_outcomes=["the order is placed"],
            setup="preserve",
        )

    def _run(self, replay):
        found = ScreenAssessment(
            screen_name="Checkout",
            detected_issues=[
                QAIssue(
                    issue_type="visual_glitch",
                    severity="low",
                    description="The total overlaps the button.",
                )
            ],
        )

        class OneLook:
            """Stands in for the auditor, which needs a model."""

            def __init__(self, auditor):
                self.given = False

            def audit_snapshot(self, snapshot):
                if self.given:
                    return []
                self.given = True
                return [found]

            def request_repeat_audit(self):
                pass

            def finalize_audit(self):
                return []

        agent = self._agent(replay)
        with patch("yewo.testing.obligation_auditing.ScreenAuditSession", OneLook):
            agent.execute_scenario(
                self._case(),
                [],
                SimpleNamespace(
                    platform=Platform.ANDROID,
                    serial="",
                    os_version=None,
                    package_id=None,
                    driver=None,
                ),
                SimpleNamespace(reset=Mock()),
                Path("unused"),
                session_trail=object(),
            )
        return found.detected_issues[0].step

    def test_the_observation_step_is_used_even_after_cached_steps(self):
        self.assertEqual(self._run(replay_of(2)), 3)


if __name__ == "__main__":
    unittest.main()
