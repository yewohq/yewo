"""Measured outcomes stay identical in every report view."""

import json
import tempfile
import unittest
from pathlib import Path

from fakes import walk

from yewo.testing.models import (
    GeneratedTestCase,
    JourneyReport,
    Measurement,
    RuleVerification,
    RunEnvironment,
    ScenarioReport,
)
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject
from yewo.testing.report.page import document, page_of_record
from yewo.testing.report.reports import save_junit_report, save_markdown_report


class MeasuredOutcomeArtifactTests(unittest.TestCase):
    def test_every_artifact_exposes_the_same_measured_outcome(self):
        statement = "The app launches within its declared limit."
        reasoning = "Worst sample was 900 ms; required at most 800 ms."
        case = GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Launch",
            cell="performance.happy_path.0",
            category="performance",
            scenario_type="happy_path",
            priority="P1",
            steps=walk("Open the app."),
            expected_outcomes=[
                {
                    "statement": statement,
                    "metric": "app_launch_ttid_ms",
                    "threshold": 800,
                }
            ],
        )
        scenario = ScenarioReport(
            scenario_uuid=case.scenario_uuid,
            test_id=case.test_id,
            scenario_name=case.scenario_name,
            cell=case.cell,
            status="FAILED",
            failure_reason="Rule validation failed.",
            measurements=[
                Measurement(
                    metric="app_launch_ttid_ms",
                    value=900,
                    unit="ms",
                    source="device",
                    collector="Android Activity Manager am start -S -W",
                    platform="android",
                    target_kind="emulator",
                    os_version="35",
                    device_model="Pixel 8",
                    app_data_state="clear_data",
                    launch_type="cold_process",
                    iteration=1,
                    collected_at="2026-10-25T12:00:00Z",
                    scenario_uuid=case.scenario_uuid,
                )
            ],
            rule_verifications=[
                RuleVerification(
                    rule=statement,
                    rule_id="expected:0",
                    status="failed",
                    reasoning=reasoning,
                )
            ],
        )
        environment = RunEnvironment(
            yewo_version="3.0.0",
            model="test/model",
            platform="ios",
            grounding="marks",
        )
        report = JourneyReport(
            journey_id="launch",
            journey_name="Launch",
            description="Launch the app.",
            status="FAILED",
            scenarios=[scenario],
            failed_count=1,
            environment=environment,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            markdown = save_markdown_report(report, output).read_text(encoding="utf-8")
            junit = save_junit_report(report, output).read_text(encoding="utf-8")
            manifest = RunManifest(
                output / "run.json",
                RunSubject(kind="journey", id="launch", name="Launch"),
                [case],
                environment,
                RunInputs(subject="launch", subject_sha256="a" * 64),
            )
            manifest.record(case.test_id, scenario)
            manifest.finish("FAILED")
            record = json.loads(manifest.path.read_text(encoding="utf-8"))
            html = document(page_of_record(record, generated="25 Oct 2026, 12:00 UTC"))

        for artifact in (markdown, junit, json.dumps(record), html):
            self.assertIn(statement, artifact)
            self.assertIn(reasoning, artifact)
            self.assertIn("app_launch_ttid_ms", artifact)
            self.assertIn("900", artifact)

        for canonical_artifact in (junit, json.dumps(record)):
            self.assertIn("expected:0", canonical_artifact)


if __name__ == "__main__":
    unittest.main()
