"""Recovery transactions survive the canonical report without leaking values."""

import json
import tempfile
from pathlib import Path

from fakes import walk

from yewo.agent.run.recovery import RecoveryCause, RecoveryMode, RecoveryTransaction
from yewo.testing.models import (
    GeneratedTestCase,
    RunEnvironment,
    ScenarioReport,
)
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject
from yewo.utils.secrets import SecretScrubber


def test_recovery_evidence_is_scrubbed_into_the_canonical_record():
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"account": "private-value"})
    recovery = RecoveryTransaction(
        failed_step=1,
        opened_step=2,
        original_intent="Enter private-value",
        failure_reason="private-value was not entered",
        cause=RecoveryCause.GROUNDING_DRIFT,
        mode=RecoveryMode.MODEL_PROPOSAL,
    )
    recovery.recover("private-value appeared in the original field")
    scenario = GeneratedTestCase(
        test_id="TS_001",
        scenario_name="Recovery",
        cell="functional.happy_path.0",
        category="functional",
        scenario_type="happy_path",
        priority="P0",
        steps=walk("Enter the account."),
        expected_outcomes=["The account is entered."],
    )
    report = ScenarioReport(
        test_id=scenario.test_id,
        scenario_name=scenario.scenario_name,
        cell=scenario.cell,
        status="FAILED",
        recovery_transactions=[recovery],
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.json"
        manifest = RunManifest(
            path,
            RunSubject(kind="journey", id="j", name="Journey"),
            [scenario],
            environment=RunEnvironment(
                yewo_version="0.1.0",
                model="model",
                platform="android",
                grounding="marks",
            ),
            inputs=RunInputs(subject="j", subject_sha256="0" * 64),
            scrubber=scrubber,
        )
        manifest.record(scenario.test_id, report)
        stored = json.loads(path.read_text())["scenarios"][0]["recovery_transactions"]

    assert "private-value" not in json.dumps(stored)
    assert "{{account}}" in json.dumps(stored)
