"""Runtime controller evidence survives the canonical run record."""

import json
from pathlib import Path

from yewo.agent.memory.control_identity import ControlIdentity
from yewo.agent.run.input_transaction import (
    FocusReceipt,
    InputTransactionReceipt,
    InputTransactionStatus,
)
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.testing.models import GeneratedTestCase, RunEnvironment, ScenarioReport
from yewo.testing.obligations import (
    ObligationEvent,
    ObligationReadiness,
    ObligationRouteReceipt,
    ObligationRunEvidence,
    RouteDecision,
)
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject


def test_runtime_contract_receipts_are_persisted_in_run_json(tmp_path: Path) -> None:
    scenario = GeneratedTestCase(
        test_id="TS_001",
        scenario_name="Scenario",
        cell="functional.happy_path.0",
        category="functional",
        scenario_type="happy_path",
        priority="P0",
        steps=[{"kind": "action", "instruction": "Open the app."}],
        expected_outcomes=["The app opens."],
    )
    manifest = RunManifest(
        tmp_path / "run.json",
        RunSubject(kind="journey", id="j1", name="Journey"),
        [scenario],
        RunEnvironment(
            yewo_version="0.1", model="model", platform="android", grounding="marks"
        ),
        RunInputs(subject="j1", subject_sha256="0" * 64),
    )
    report = ScenarioReport(
        test_id="TS_001",
        scenario_name="Scenario",
        cell="functional.happy_path.0",
        status="ABORTED",
        settlement_receipts=[
            SettlementReceipt(
                settlement_id="settlement-1",
                control_identity=ControlIdentity(
                    "tap", "0" * 64, "relative", ((0.5, 0.5),), "1:100"
                ),
                source_phash="0" * 64,
                destination_phash="f" * 64,
                status=SettlementStatus.UNVERIFIED,
                reason="Another action began before settlement.",
            )
        ],
        input_transactions=[
            InputTransactionReceipt(
                transaction_id="input-1",
                value_name="schedule_name",
                status=InputTransactionStatus.CONTEXT_LOST,
                value_verified=False,
                context_retained=False,
                reason="The form closed.",
                focus=FocusReceipt(keyboard_after=True),
            )
        ],
        obligation_evidence=ObligationRunEvidence(
            routes=[
                ObligationRouteReceipt(
                    obligation_id="outcome-1",
                    event=ObligationEvent.RUN_END,
                    decision=RouteDecision.ACTIVE,
                )
            ],
            readiness=ObligationReadiness(1, 1, 0, 0, 0),
        ),
    )

    manifest.record("TS_001", report)
    stored = json.loads(manifest.path.read_text())["scenarios"][0]

    assert stored["settlement_receipts"][0]["settlement_id"] == "settlement-1"
    assert stored["input_transactions"][0]["status"] == "context_lost"
    assert stored["obligation_evidence"]["routes"][0]["decision"] == "active"
