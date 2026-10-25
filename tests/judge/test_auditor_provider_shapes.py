"""Equivalent provider labels must not abort an otherwise useful audit."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
