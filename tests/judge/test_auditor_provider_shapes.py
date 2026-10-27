"""Equivalent provider labels must not abort an otherwise useful audit."""

import unittest

from pydantic import ValidationError

from yewo.testing.judge.qa_auditor import ScreenAssessment


class ProviderShapeNormalizationTests(unittest.TestCase):
    def test_a_sensitive_subject_names_the_finding_field(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Login",
                "sensitive_findings": [
                    {"subject": "password", "rendering": "label_only"}
                ],
            }
        )

        self.assertEqual(assessment.sensitive_findings[0].field, "password")

    def test_null_or_missing_field_in_sensitive_finding_is_normalized(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Login",
                "sensitive_findings": [
                    {"field": None, "rendering": "absent"},
                    {"rendering": "absent"},
                ],
            }
        )

        self.assertEqual(assessment.sensitive_findings[0].field, "unspecified")
        self.assertEqual(assessment.sensitive_findings[1].field, "unspecified")

    def test_an_unranked_visible_issue_is_kept_as_medium(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Home",
                "detected_issues": [
                    {"issue_type": "overlap", "description": "Cards overlap."}
                ],
            }
        )

        issue = assessment.detected_issues[0]
        self.assertEqual(issue.severity, "medium")
        self.assertEqual(issue.description, "Cards overlap.")

    def test_a_plain_issue_sentence_is_kept_as_an_unranked_finding(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Login",
                "detected_issues": ["The bottom of the form exceeds its bounds."],
            }
        )

        issue = assessment.detected_issues[0]
        self.assertEqual(issue.issue_type.value, "other")
        self.assertEqual(issue.severity, "medium")
        self.assertEqual(issue.description, "The bottom of the form exceeds its bounds.")

    def test_issue_type_and_reason_aliases_are_normalized(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Login",
                "detected_issues": [
                    {
                        "type": "overflow",
                        "reason": "The form exceeds its bounds.",
                        "severity": "medium",
                    }
                ],
            }
        )

        issue = assessment.detected_issues[0]
        self.assertEqual(issue.issue_type.value, "other")
        self.assertEqual(issue.description, "The form exceeds its bounds.")

    def test_rule_aliases_are_normalized_by_the_rule_contract(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Cart",
                "rule_verifications": [
                    {
                        "id": "expected:0",
                        "statement": "The cart is empty.",
                        "verdict": "pass",
                        "reason": "The empty state is visible.",
                    }
                ],
            }
        )

        verification = assessment.rule_verifications[0]
        self.assertEqual(verification.rule_id, "expected:0")
        self.assertEqual(verification.status, "passed")
        self.assertEqual(verification.reasoning, "The empty state is visible.")

    def test_not_applicable_status_cannot_default_to_applicable(self):
        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Home",
                "rule_verifications": [
                    {
                        "rule": "The password is masked.",
                        "status": "not applicable",
                        "reasoning": "There is no password field.",
                    }
                ],
            }
        )

        verification = assessment.rule_verifications[0]
        self.assertEqual(verification.status, "untested")
        self.assertFalse(verification.applicable)

    def test_missing_reasoning_is_rejected_instead_of_invented(self):
        with self.assertRaises(ValidationError):
            ScreenAssessment.model_validate(
                {
                    "screen_name": "Home",
                    "rule_verifications": [
                        {"rule": "Home is visible.", "status": "passed"}
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
