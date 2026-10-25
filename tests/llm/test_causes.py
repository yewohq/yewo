"""Whose problem a scenario that reached no verdict is.

The status says what happened. A red build asks a different question: what do
I do about it. A dead emulator, an exhausted API key and a thin audit all end
a scenario, and all three want a different answer.
"""

import unittest
from types import SimpleNamespace

from fakes import device_result

from yewo.errors import (
    ArtifactWriteFailed,
    Cause,
    DeviceUnavailable,
    ObservationUnavailable,
    cause_of,
)


class ClassifyTests(unittest.TestCase):
    """Read from the exception's type, never from its words."""

    def test_a_device_yewo_cannot_drive(self):
        self.assertIs(cause_of(DeviceUnavailable("offline")), Cause.DEVICE)
        self.assertIs(cause_of(ObservationUnavailable("no screen")), Cause.DEVICE)

    def test_a_driver_that_lost_its_session(self):
        """Appium ends a session that sends no command while a step thinks."""
        from selenium.common.exceptions import WebDriverException

        self.assertIs(cause_of(WebDriverException("terminated")), Cause.DEVICE)

    def test_a_model_service_that_refused(self):
        """A key out of credit, a rate limit, a timeout: the service, not the app."""
        import litellm

        failure = litellm.exceptions.APIError(403, "Key limit exceeded", "openrouter", "m")

        self.assertIs(cause_of(failure), Cause.PROVIDER)

    def test_an_artifact_yewo_could_not_write(self):
        self.assertIs(cause_of(ArtifactWriteFailed("read-only")), Cause.ARTIFACT)

    def test_anything_unrecognised_claims_nothing(self):
        """Naming a cause without evidence is how a report lies."""
        self.assertIs(cause_of(ValueError("surprise")), Cause.UNKNOWN)
        self.assertIs(cause_of(KeyError("surprise")), Cause.UNKNOWN)

    def test_the_message_is_never_read(self):
        """A message is written for a person and changes between versions."""
        self.assertIs(cause_of(ValueError("Error executing adbExec: offline")), Cause.UNKNOWN)


class VerdictCauseTests(unittest.TestCase):
    """Every answer that is not PASSED or FAILED says whose problem it is."""

    def _verdict(self, **overrides):
        from yewo.testing.qa_agent import QAAgent

        values = {
            "run_result": device_result(),
            "stop_reason": None,
            "all_verifications": [],
            "all_issues": [],
            "qa_outputs": [
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            "assessments": [SimpleNamespace(rule_verifications=[], detected_issues=[])],
            "required_outcomes": ["Order is confirmed"],
            "security_rules": [],
        }
        values.update(overrides)
        return QAAgent.__new__(QAAgent)._determine_status(**values)

    def _run(self, status, cause=None):
        return device_result(status=status, failure_reason="why", cause=cause)

    def test_an_error_carries_the_cause_the_run_recorded(self):
        status, _reason, cause = self._verdict(
            run_result=self._run("exception", Cause.PROVIDER)
        )

        self.assertEqual(status, "ERROR")
        self.assertIs(cause, Cause.PROVIDER)

    def test_an_error_with_nothing_recorded_claims_nothing(self):
        status, _reason, cause = self._verdict(run_result=self._run("exception"))

        self.assertEqual(status, "ERROR")
        self.assertIs(cause, Cause.UNKNOWN)

    def test_a_run_that_never_settled_is_the_agent(self):
        """Yewo drove the app and never reached a state worth judging."""
        for status in ("loop_detected", "step_limit"):
            _status, _reason, cause = self._verdict(run_result=self._run(status))

            self.assertIs(cause, Cause.AGENT, status)

    def test_missing_evidence_is_the_evidence(self):
        status, reason, cause = self._verdict()

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)
        self.assertIs(cause, Cause.EVIDENCE)

    def test_a_scenario_that_asked_the_unanswerable_is_the_scenario(self):
        from yewo.testing.models import AssertionScope, ExpectedOutcome

        _status, _reason, cause = self._verdict(
            required_outcomes=[
                ExpectedOutcome(
                    statement="the app stays responsive", scope=AssertionScope.RUN_TIMING
                )
            ]
        )

        self.assertIs(cause, Cause.SCENARIO)

    def test_a_scenario_with_nothing_to_check_is_the_scenario(self):
        _status, _reason, cause = self._verdict(required_outcomes=[])

        self.assertIs(cause, Cause.SCENARIO)

    def test_an_infeasible_scenario_is_the_scenario(self):
        _status, _reason, cause = self._verdict(
            qa_outputs=[
                SimpleNamespace(
                    is_scenario_feasible=False, infeasibility_reason="no such screen"
                )
            ]
        )

        self.assertIs(cause, Cause.SCENARIO)

    def test_a_pass_needs_no_cause(self):
        from yewo.testing.models import RuleVerification

        status, _reason, cause = self._verdict(
            all_verifications=[
                RuleVerification(
                    rule="expected:0", status="passed", reasoning="seen",
                    screen_name="Order Status",
                )
            ]
        )

        self.assertEqual(status, "PASSED")
        self.assertIsNone(cause)

    def test_a_failure_needs_no_cause(self):
        """The app is the answer, so there is nothing to attribute."""
        status, _reason, cause = self._verdict(
            all_issues=[
                SimpleNamespace(
                    issue_type="crash_screen", severity="critical",
                    description="crashed", current_screen="Home",
                )
            ]
        )

        self.assertEqual(status, "FAILED")
        self.assertIsNone(cause)


class RecordedCauseTests(unittest.TestCase):
    """The run records its cause, and the gate reports it unchanged."""

    def test_the_run_records_a_cause_and_not_a_string(self):
        """Typed on DeviceRunResult, so no reader converts or guesses."""
        from yewo.agent.run.outcomes import DeviceRunResult

        field = DeviceRunResult.__dataclass_fields__["cause"]

        self.assertIn("Cause", str(field.type))

    def test_the_gate_reports_the_cause_the_run_recorded(self):
        from yewo.testing.judge.verdicts import _terminal_outcome

        status, _reason, cause = _terminal_outcome(
            run_result=device_result(
                status="exception", failure_reason="boom", cause=Cause.DEVICE
            ),
            stop_reason=None,
            verifications=[],
            issues=[],
            outputs=[],
        )

        self.assertEqual(status, "ERROR")
        self.assertIs(cause, Cause.DEVICE)

    def test_a_run_that_recorded_nothing_reports_unknown(self):
        from yewo.testing.judge.verdicts import _terminal_outcome

        _status, _reason, cause = _terminal_outcome(
            run_result=device_result(status="exception", failure_reason="boom"),
            stop_reason=None,
            verifications=[],
            issues=[],
            outputs=[],
        )

        self.assertIs(cause, Cause.UNKNOWN)

    def test_an_unrecognised_name_is_still_a_mistake(self):
        """Nothing stores a cause as text, so leniency would only hide typos."""
        with self.assertRaises(ValueError):
            Cause("some_future_cause")
