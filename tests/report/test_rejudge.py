"""Deciding a stored run's verdicts again, from the record alone."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fakes import walk

from yewo.errors import UsageError
from yewo.testing import rejudge


def _written(directory: Path, scenarios: list, schema: int = 2) -> Path:
    path = directory / "run.json"
    path.write_text(
        json.dumps({"schema": schema, "run_uuid": "r1", "scenarios": scenarios}),
        encoding="utf-8",
    )
    return path


def _scenario(**overrides) -> dict:
    scenario = {
        "test_id": "TS_functional_happy_path_0",
        "cell": "functional.happy_path.0",
        "scenario_uuid": "s1",
        "status": "PASSED",
        "rule_verifications": [
            {
                "rule": "Order is confirmed",
                "rule_id": "expected:0",
                "status": "passed",
                "reasoning": "It says confirmed.",
                "screen_name": "Confirmation",
            }
        ],
        "detected_issues": [],
        "verdict_inputs": {
            "run_status": "goal_reached",
            "screens_judged": 1,
            "expected_outcomes": [
                {"statement": "Order is confirmed", "scope": "current_screen"}
            ],
        },
    }
    scenario.update(overrides)
    return scenario


class RejudgingTests(unittest.TestCase):
    """The judge, asked what it would answer about a run that already ran."""

    def _verdicts(self, scenarios: list, schema: int = 2) -> list:
        """Every scenario decided again, or the refusal it raised."""
        with TemporaryDirectory() as directory:
            written, entries = rejudge.scenarios_in(
                _written(Path(directory), scenarios, schema)
            )
        decided = []
        for entry in entries:
            try:
                decided.append(rejudge.verdict_of(entry, written))
            except rejudge.UnreadableRunRecord as refused:
                decided.append(refused)
        return decided

    def test_a_scenario_reaches_the_verdict_it_recorded(self):
        self.assertEqual(self._verdicts([_scenario()])[0].status, "PASSED")

    def test_the_recorded_word_is_never_what_the_judge_answers(self):
        """A record says what happened. Deciding it again asks about the code.

        The entry records FAILED and nothing in it supports one, so the judge
        answers PASSED. The file is not touched either way.
        """
        self.assertEqual(self._verdicts([_scenario(status="FAILED")])[0].status, "PASSED")

    def test_every_input_the_judge_takes_is_read_back(self):
        """A run that declared a state the device never confirmed stays
        unverified when it is decided again."""
        decided = self._verdicts([
            _scenario(
                status="UNVERIFIED",
                verdict_inputs={
                    "run_status": "goal_reached",
                    "screens_judged": 1,
                    "expected_outcomes": [
                        {"statement": "Order is confirmed", "scope": "current_screen"}
                    ],
                    "required_capabilities": ["network"],
                    "receipts": [{"capability": "network", "confirmed": False}],
                },
            )
        ])

        self.assertEqual(decided[0].status, "UNVERIFIED")
        self.assertIn("never confirmed", decided[0].reason)

    def test_the_planners_last_word_is_read_back(self):
        decided = self._verdicts([
            _scenario(
                status="SKIPPED",
                verdict_inputs={
                    "run_status": "goal_reached",
                    "screens_judged": 1,
                    "is_scenario_feasible": False,
                    "infeasibility_reason": "This build has no guest checkout.",
                },
            )
        ])

        self.assertEqual(decided[0].status, "SKIPPED")
        self.assertIn("guest checkout", decided[0].reason)

    def test_a_finding_that_contradicts_an_exposure_claim_is_read_back(self):
        """The reading is the evidence, and it has to survive into the record."""
        decided = self._verdicts([
            _scenario(
                detected_issues=[{
                    "issue_type": "sensitive_data_exposure",
                    "severity": "critical",
                    "description": "The CVV looks like plain text.",
                    "current_screen": "Payment",
                }],
                verdict_inputs={
                    "run_status": "goal_reached",
                    "screens_judged": 1,
                    "expected_outcomes": [
                        {"statement": "Order is confirmed", "scope": "current_screen"}
                    ],
                    "sensitive_findings": [
                        {"field": "CVV", "rendering": "label_only",
                         "screen_name": "Payment"}
                    ],
                },
            )
        ])

        self.assertEqual(decided[0].status, "PASSED")

    def test_a_record_written_before_the_inputs_were_kept_is_refused(self):
        """Half of what the judge was given reaches a verdict that is not one."""
        refused = self._verdicts([_scenario()], schema=1)[0]

        self.assertIsInstance(refused, rejudge.UnreadableRunRecord)
        self.assertIn("schema 1", str(refused))

    def test_a_scenario_that_never_reached_a_judge_is_refused(self):
        refused = self._verdicts([_scenario(verdict_inputs=None, status="ERROR")])[0]

        self.assertIsInstance(refused, rejudge.UnreadableRunRecord)

    def test_a_file_that_is_not_a_record_is_refused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text('{"something": "else"}', encoding="utf-8")

            with self.assertRaises(UsageError) as refused:
                rejudge.scenarios_in(path)

        self.assertIn("does not hold a run record", str(refused.exception))

    def test_a_file_that_is_absent_is_refused(self):
        with TemporaryDirectory() as directory, self.assertRaises(UsageError):
            rejudge.scenarios_in(Path(directory) / "absent.json")

    def test_malformed_json_is_refused_rather_than_read_as_empty(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(UsageError) as refused:
                rejudge.scenarios_in(path)

        self.assertIn("not readable as JSON", str(refused.exception))


class WhatTheReportCarriesTests(unittest.TestCase):
    """The record is only as good as what the agent writes into it."""

    def _agent(self):
        from types import SimpleNamespace

        from yewo.testing.qa_agent import QAAgent
        from yewo.utils.secrets import SecretScrubber

        agent = QAAgent.__new__(QAAgent)
        agent._device_agent = SimpleNamespace(scrubber=SecretScrubber())
        return agent

    def _case(self, outcomes=("Order is confirmed",), capabilities=()):
        from yewo.testing.models import ExpectedOutcome, GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_001",
            scenario_name="Sign in",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Enter the email.", "Tap Login."),
            expected_outcomes=[ExpectedOutcome(statement=one) for one in outcomes],
            required_capabilities=list(capabilities),
        )

    def _assessment(self, findings=(), issues=()):
        from yewo.testing.judge.qa_auditor import ScreenAssessment
        from yewo.testing.models import RuleVerification

        return ScreenAssessment(
            screen_name="Payment",
            rule_verifications=[
                RuleVerification(
                    rule="Order is confirmed", rule_id="expected:0",
                    status="passed", reasoning="It says confirmed.",
                )
            ],
            sensitive_findings=list(findings),
            detected_issues=list(issues),
        )

    def _result(self, status="goal_reached", receipts=()):
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.run.outcomes import DeviceRunResult

        return DeviceRunResult(
            status=status, steps_executed=2, failure_reason=None,
            history=AgentHistory(), capability_receipts=list(receipts),
        )

    def _reported(self, assessment=None, case=None, result=None, outputs=None):
        from types import SimpleNamespace

        return self._agent()._build_report(
            test_case=case or self._case(),
            run_result=result or self._result(),
            qa_outputs=outputs or [
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            assessments=[assessment or self._assessment()],
            stop_reason=None,
            duration=1.0,
            security_rules=["The card number is masked"],
        )

    def test_a_report_carries_what_its_verdict_was_decided_from(self):
        report = self._reported()
        inputs = report.verdict_inputs

        self.assertIsNotNone(inputs)
        self.assertEqual(inputs.run_status, "goal_reached")
        self.assertEqual(inputs.screens_judged, 1)
        self.assertEqual(inputs.security_rules, ["The card number is masked"])
        self.assertEqual(
            [outcome.statement for outcome in inputs.expected_outcomes],
            ["Order is confirmed"],
        )

    def test_the_findings_reach_the_verdict_and_not_only_the_record(self):
        """The auditor answers twice about a sensitive field, and the reading
        is the evidence. A claim the reading contradicts must not fail the run.

        The findings were collected and never handed to the judge, so this
        passed on the fixtures and failed on every real screen.
        """
        from yewo.testing.models import QAIssue, Rendering, SensitiveFinding

        report = self._reported(
            assessment=self._assessment(
                findings=[SensitiveFinding(field="CVV", rendering=Rendering.LABEL_ONLY)],
                issues=[QAIssue(
                    issue_type="sensitive_data_exposure", severity="critical",
                    description="The CVV looks like plain text.",
                )],
            )
        )

        self.assertEqual(report.status, "PASSED")
        self.assertEqual(len(report.verdict_inputs.sensitive_findings), 1)

    def test_a_declared_capability_and_its_receipt_are_both_carried(self):
        from yewo.agent.actuation.actions import CapabilityKind
        from yewo.agent.device.receipts import Receipt

        report = self._reported(
            case=self._case(capabilities=[CapabilityKind.NETWORK]),
            result=self._result(receipts=[Receipt(
                capability="network", requested="offline", observed=None,
                target="a device", source="adb",
            )]),
        )
        inputs = report.verdict_inputs

        self.assertEqual(inputs.required_capabilities, [CapabilityKind.NETWORK])
        self.assertEqual(inputs.receipts[0].capability, "network")
        self.assertFalse(inputs.receipts[0].confirmed)

    def test_the_planners_last_word_is_carried(self):
        from types import SimpleNamespace

        report = self._reported(outputs=[
            SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None),
            SimpleNamespace(
                is_scenario_feasible=False,
                infeasibility_reason="This build has no guest checkout.",
            ),
        ])

        self.assertFalse(report.verdict_inputs.is_scenario_feasible)
        self.assertEqual(report.status, "SKIPPED")

    def test_a_report_reaches_its_own_verdict_when_it_is_decided_again(self):
        """The whole point: the record holds what its verdict came from."""
        report = self._reported()
        entry = json.loads(report.model_dump_json())
        entry["cell"] = report.cell

        with TemporaryDirectory() as directory:
            schema, entries = rejudge.scenarios_in(_written(Path(directory), [entry]))

        self.assertEqual(rejudge.verdict_of(entries[0], schema).status, report.status)


if __name__ == "__main__":
    unittest.main()
