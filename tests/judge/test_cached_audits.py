"""Live audits keep their own evidence and recheck transient launch frames."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.actuation.actions import Coordinates, TapAction, TypeAction
from yewo.agent.run.outcomes import ObservationDirective
from yewo.testing.judge.qa_auditor import QAAuditor, ScreenAssessment, ScreenAuditSession
from yewo.testing.judge.screen_audits import LiveScreenAudits
from yewo.testing.models import QAIssue, RuleVerification


def assessment(crash=False):
    issues = []
    if crash:
        issues.append(
            QAIssue(
                issue_type="crash_screen",
                severity="critical",
                description="Blank screen",
            )
        )
    return ScreenAssessment(screen_name="Screen", detected_issues=issues)


def snapshot(phash, reason):
    return SimpleNamespace(
        raw_img_bytes=b"png",
        current_phash=phash,
        scan_info=SimpleNamespace(reason=reason),
        gui_elements=[],
    )


class ScreenImageTests(unittest.TestCase):
    def test_the_policy_contains_no_example_app_domain(self):
        from yewo.testing.judge.qa_auditor import PROMPT

        for domain_term in (
            "payment",
            "card number",
            "checkout",
            "place order",
            "account number",
            "sign-in",
        ):
            self.assertNotIn(domain_term, PROMPT.lower())

    def test_transport_masks_are_not_app_defects(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=assessment()))
        QAAuditor(llm, [], []).audit(b"png", "screen")
        message = llm.generate_response.call_args.args[0][0]
        prompt = message.to_dict()["content"]

        self.assertIn("`yewo_secret_mask`", prompt)
        self.assertIn("transport metadata, not app content", prompt)
        self.assertIn("Do not treat it\nas a button", prompt)
        self.assertIn("blank screen, or\ncrash", prompt)

    def test_each_look_uses_the_image_prepared_for_that_reading(self):
        sent = []

        class Recording:
            state = b"arrival"

            def prepare_model_image(self, screenshot, gui_elements=()):
                return self.state

            def audit(
                self,
                screenshot,
                screen_id,
                gui_elements=(),
                prepared_model_image=None,
            ):
                sent.append((screen_id, prepared_model_image))
                return assessment()

        auditor = Recording()
        session = ScreenAuditSession(auditor)
        session.audit_snapshot(snapshot("A", "first_pass"))
        auditor.state = b"filled"
        session.audit_snapshot(snapshot("A-filled", "stable"))
        auditor.state = b"next-screen"

        session.audit_snapshot(snapshot("B", "navigation"))

        self.assertEqual(
            sent,
            [("A", b"arrival"), ("A-filled", b"filled"), ("B", b"next-screen")],
        )


class LiveLaunchTests(unittest.TestCase):
    @staticmethod
    def _flow(answers):
        class Recording:
            def audit(self, *args, **kwargs):
                return answers.pop(0)

        return LiveScreenAudits(ScreenAuditSession(Recording()))

    def test_a_launch_crash_forces_a_wait_and_a_fresh_look(self):
        flow = self._flow([assessment(crash=True), assessment()])

        directive = flow.observe(snapshot("launch", "first_pass"), 1)
        response = SimpleNamespace(action=None, is_scenario_feasible=False)

        self.assertIs(directive, ObservationDirective.PLAN)
        self.assertTrue(flow.apply_launch_recheck(response))
        self.assertTrue(response.is_scenario_feasible)
        self.assertEqual(response.action.type, "wait")
        self.assertEqual(flow.assessments[0].detected_issues, [])
        self.assertIs(
            flow.observe(snapshot("checkout", "navigation"), 2),
            ObservationDirective.CONTINUE,
        )

    def test_a_crash_confirmed_on_the_second_look_stops_before_action(self):
        flow = self._flow([assessment(crash=True), assessment(crash=True)])

        flow.observe(snapshot("launch", "first_pass"), 1)
        directive = flow.observe(snapshot("launch", "stable"), 2)

        self.assertIs(directive, ObservationDirective.STOP)
        self.assertEqual(flow.stop_reason, "App crashed: crash screen detected.")
        self.assertEqual(flow.assessments[-1].detected_issues[0].step, 2)

    def test_a_failed_expected_outcome_is_decided_after_the_journey(self):
        failed = assessment()
        failed.rule_verifications = [
            RuleVerification(
                rule="The declared result is shown",
                rule_id="expected:0",
                status="failed",
                reasoning="The screen shows a contradictory result.",
            )
        ]
        flow = self._flow([failed])

        directive = flow.observe(snapshot("result", "first_pass"), 1)

        self.assertIs(directive, ObservationDirective.CONTINUE)
        self.assertIsNone(flow.stop_reason)

    def test_a_failed_security_rule_stops_before_a_dependent_action(self):
        failed = assessment()
        failed.rule_verifications = [
            RuleVerification(
                rule="The secret is never exposed",
                rule_id="security:0",
                status="failed",
                reasoning="The screen shows the secret in plain text.",
            )
        ]
        flow = self._flow([failed])

        directive = flow.observe(snapshot("result", "first_pass"), 1)

        self.assertIs(directive, ObservationDirective.STOP)
        self.assertIn("secret is never exposed", flow.stop_reason)

    def test_an_observed_state_is_not_reaudited_before_a_tap(self):
        flow = self._flow([assessment(), assessment()])
        flow.observe(snapshot("form", "first_pass"), 1)
        flow.observe(snapshot("form-filled", "stable"), 2)

        directive = flow.before_action(
            TapAction(
                coordinates=Coordinates(x=10, y=10),
                target="Continue",
                reason="Continue.",
            ),
            2,
        )

        self.assertIs(directive, ObservationDirective.CONTINUE)
        self.assertEqual(len(flow.assessments), 2)

    def test_an_observed_state_is_audited_before_another_entry(self):
        flow = self._flow([assessment(), assessment()])
        flow.observe(snapshot("form", "first_pass"), 1)
        flow.observe(snapshot("form-filled", "stable"), 2)

        directive = flow.before_action(
            TypeAction(
                text="next",
                coordinates=Coordinates(x=10, y=10),
                target="Next field",
                reason="Fill the form.",
            ),
            2,
        )

        self.assertIs(directive, ObservationDirective.CONTINUE)
        self.assertEqual(len(flow.assessments), 2)


if __name__ == "__main__":
    unittest.main()
