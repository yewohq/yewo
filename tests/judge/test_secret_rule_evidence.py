"""Local evidence must describe the app, not Yewo's transport redaction."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.actuation.actions import Coordinates
from yewo.testing.judge.qa_auditor import QAAuditor, ScreenAssessment
from yewo.testing.models import RuleVerification, SensitiveFinding
from yewo.utils.secrets import AppValueRendering, SecretScrubber


class SecretRuleEvidenceTests(unittest.TestCase):
    def _scrubber(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"account_secret": "private-value"})
        return scrubber

    def _passed(self, finding):
        return ScreenAssessment(
            screen_name="Account form",
            rule_verifications=[
                RuleVerification(
                    rule="The account secret stays masked",
                    rule_id="security:0",
                    status="passed",
                    reasoning="No readable value is visible.",
                )
            ],
            sensitive_findings=[finding],
        )

    def _audit(self, scrubber, answer):
        llm = SimpleNamespace(generate_response=Mock(return_value=answer))
        return QAAuditor(
            llm,
            [],
            ["The account_secret stays masked"],
            scrubber=scrubber,
        ).audit(b"png", "screen", prepared_model_image=b"safe")

    def test_an_empty_declared_field_cannot_pass_a_security_rule(self):
        answer = self._passed(
            SensitiveFinding(field="Account Secret", rendering="label_only")
        )

        assessment = self._audit(self._scrubber(), answer)

        self.assertEqual(assessment.rule_verifications[0].status, "untested")

    def test_yewo_transport_mask_alone_cannot_pass_a_security_rule(self):
        scrubber = self._scrubber()
        scrubber.protect_typed_region(
            "$ACCOUNT_SECRET",
            Coordinates(x=20, y=20),
            [],
        )
        answer = self._passed(
            SensitiveFinding(
                field="account_secret",
                rendering="masked",
                quoted="$ACCOUNT_SECRET",
            )
        )

        assessment = self._audit(scrubber, answer)

        self.assertEqual(assessment.rule_verifications[0].status, "untested")

    def test_local_app_masking_can_support_the_auditors_pass(self):
        scrubber = self._scrubber()
        scrubber.protect_typed_bounds(
            "private-value",
            (5, 5, 70, 30),
            AppValueRendering.MASKED,
        )
        answer = self._passed(
            SensitiveFinding(field="account_secret", rendering="masked")
        )

        assessment = self._audit(scrubber, answer)

        self.assertEqual(assessment.rule_verifications[0].status, "passed")


if __name__ == "__main__":
    unittest.main()
