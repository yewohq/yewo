"""Run-level policy checks remain visible and consistent in every artifact."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import ValidationError

from fakes import walk

from yewo.testing.models import (
    GeneratedTestCase,
    JourneyReport,
    PolicyCheckReport,
    RunEnvironment,
    ScenarioReport,
)
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject
from yewo.testing.report.page import RUN_FILES, document, page_of_record
from yewo.testing.report.page.view import page_of
from yewo.testing.report.reports import save_junit_report, save_markdown_report
from yewo.utils.secrets import SecretScrubber


def policy_check(status: str = "passed", **fields) -> PolicyCheckReport:
    values = {
        "check_id": "android.debuggable",
        "status": status,
        "platform": "android",
        "source": "aapt dump xmltree app.apk AndroidManifest.xml",
        "evidence": "android:debuggable=false",
    }
    values.update(fields)
    return PolicyCheckReport(**values)


def environment() -> RunEnvironment:
    return RunEnvironment(
        yewo_version="3.0.0",
        model="test/model",
        platform="android",
        grounding="marks",
    )


def scenario() -> tuple[GeneratedTestCase, ScenarioReport]:
    case = GeneratedTestCase(
        test_id="TS_001",
        scenario_name="Checkout",
        cell="functional.happy_path.0",
        category="functional",
        scenario_type="happy_path",
        priority="P0",
        steps=walk("Open checkout."),
        expected_outcomes=["Checkout is visible."],
    )
    report = ScenarioReport(
        scenario_uuid=case.scenario_uuid,
        test_id=case.test_id,
        scenario_name=case.scenario_name,
        cell=case.cell,
        status="PASSED",
    )
    return case, report


class PolicyCheckContractTests(unittest.TestCase):
    def test_a_positional_check_id_is_refused(self):
        with self.assertRaises(ValidationError):
            policy_check(check_id="baseline:0")

    def test_a_waiver_requires_a_reason(self):
        with self.assertRaises(ValidationError):
            policy_check(status="waived")

        waived = policy_check(
            status="waived",
            waiver_reason="Required for an offline cache; TICKET-123.",
        )

        self.assertEqual(waived.waiver_reason, "Required for an offline cache; TICKET-123.")

    def test_a_nonwaived_check_cannot_carry_a_waiver_reason(self):
        with self.assertRaises(ValidationError):
            policy_check(waiver_reason="This must not hide a failed check.")

    def test_policy_evidence_is_bounded(self):
        with self.assertRaises(ValidationError):
            policy_check(evidence="x" * 4097)

    def test_blank_policy_evidence_is_refused(self):
        with self.assertRaises(ValidationError):
            policy_check(evidence="   ")

    def test_passed_and_waived_checks_are_accepted_by_a_passing_run(self):
        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="PASSED",
            policy_checks=[
                policy_check(),
                policy_check(
                    check_id="android.allow_backup",
                    status="waived",
                    waiver_reason="Accepted under TICKET-123.",
                ),
            ],
        )

        self.assertEqual(report.status, "PASSED")

    def test_an_unsuccessful_policy_check_cannot_belong_to_a_passing_run(self):
        with self.assertRaises(ValidationError):
            JourneyReport(
                journey_id="j",
                journey_name="Journey",
                description="d",
                status="PASSED",
                policy_checks=[policy_check(status="failed")],
            )


class PolicyCheckArtifactTests(unittest.TestCase):
    def _report(self) -> JourneyReport:
        _, scenario_report = scenario()
        return JourneyReport(
            journey_id="checkout",
            journey_name="Checkout",
            description="Pay with the saved card.",
            status="FAILED",
            scenarios=[scenario_report],
            passed_count=1,
            policy_checks=[
                policy_check(),
                policy_check(
                    check_id="android.allow_backup",
                    status="failed",
                    evidence="password123 was exposed by the backup policy",
                ),
                policy_check(
                    check_id="android.cleartext_traffic",
                    status="waived",
                    evidence="Cleartext is enabled for a local fixture.",
                    waiver_reason="Required by password123 fixture; TICKET-123.",
                ),
                policy_check(
                    check_id="android.exported_components",
                    status="unsupported",
                    evidence="The package manifest could not be decoded.",
                ),
                policy_check(
                    check_id="android.backup_rules",
                    status="evidence_missing",
                    evidence="The referenced backup-rules file was absent.",
                ),
            ],
            environment=environment(),
        )

    def test_markdown_and_junit_show_every_check_without_exposing_secrets(self):
        report = self._report()
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "password123"})

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            markdown = save_markdown_report(report, output, scrubber=scrubber)
            junit = save_junit_report(report, output, scrubber=scrubber)
            markdown_text = markdown.read_text(encoding="utf-8")
            junit_text = junit.read_text(encoding="utf-8")
            junit_root = ET.parse(junit).getroot()

        for check in report.policy_checks:
            self.assertIn(check.check_id, markdown_text)
            self.assertIn(check.check_id, junit_text)
        self.assertNotIn("password123", markdown_text)
        self.assertNotIn("password123", junit_text)
        self.assertEqual(junit_root.attrib["tests"], "6")
        self.assertEqual(junit_root.attrib["failures"], "3")
        self.assertEqual(junit_root.attrib["skipped"], "1")

    def test_run_record_and_html_keep_the_same_policy_checks(self):
        case, scenario_report = scenario()
        report = self._report()
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "password123"})

        with tempfile.TemporaryDirectory() as directory:
            manifest = RunManifest(
                Path(directory) / "run.json",
                RunSubject(
                    kind="journey",
                    id=report.journey_id,
                    name=report.journey_name,
                    description=report.description,
                ),
                [case],
                environment(),
                RunInputs(subject="checkout", subject_sha256="a" * 64),
                scrubber=scrubber,
            )
            manifest.record(case.test_id, scenario_report)
            manifest.finish("FAILED", policy_checks=report.policy_checks)
            payload = json.loads(manifest.path.read_text(encoding="utf-8"))

        from_report = page_of(
            report,
            generated="25 Oct 2026, 12:00 UTC",
            version="3.0.0",
            artifacts=RUN_FILES,
        )
        from_record = page_of_record(payload, generated="25 Oct 2026, 12:00 UTC")

        self.assertEqual(
            [check.check_id for check in from_report.policy_checks],
            [check.check_id for check in from_record.policy_checks],
        )
        self.assertNotIn("password123", json.dumps(payload))
        self.assertIn("Policy Checks", document(from_record))
        self.assertIn("android.allow_backup", document(from_record))
        self.assertIn("View policy evidence", document(from_record))

    def test_the_canonical_record_cannot_pass_with_a_failed_check(self):
        case, scenario_report = scenario()
        with tempfile.TemporaryDirectory() as directory:
            manifest = RunManifest(
                Path(directory) / "run.json",
                RunSubject(kind="journey", id="j", name="Journey"),
                [case],
                environment(),
                RunInputs(subject="j", subject_sha256="b" * 64),
            )
            manifest.record(case.test_id, scenario_report)
            manifest.finish("PASSED", policy_checks=[policy_check(status="failed")])
            payload = json.loads(manifest.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
