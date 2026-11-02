"""Tests for action-scoped application diagnostic normalization."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.action_diagnostics import action_application_failure
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)
from yewo.devices.logs import (
    DeviceLogArtifactKind,
    DeviceLogArtifactReference,
    DeviceLogCheckpointKind,
    DeviceLogDrainReceipt,
    DeviceLogDrainState,
    DeviceLogSource,
    DeviceLogSourceReceipt,
)


def test_action_crash_becomes_concise_planner_safe_evidence() -> None:
    captured_at = datetime(2026, 11, 1, tzinfo=timezone.utc)
    artifact = DeviceLogArtifactReference(
        artifact_id="diagnostic_1",
        boundary_id="boundary_1",
        drain_id="drain_1",
        log_type="exitinfo",
        kind=DeviceLogArtifactKind.APPLICATION_DIAGNOSTIC,
        source_name="exit-info.txt",
        failure_kind=ApplicationFailureKind.CRASH,
        occurred_at=captured_at,
        platform_reason="APP CRASH(EXCEPTION)",
        platform_code="reason=4,status=0",
        sha256="0" * 64,
        path=Path("assets/device-logs/boundary_1/exit-info.txt"),
        byte_count=10,
        entry_count=1,
        redaction_policy_id="test-policy",
        source_redaction_count=0,
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    receipt = DeviceLogDrainReceipt(
        drain_id="drain_1",
        boundary_id="boundary_1",
        drain_index=1,
        checkpoint_kind=DeviceLogCheckpointKind.ACTION,
        checkpoint_id="action_1",
        started_at=captured_at,
        completed_at=captured_at,
        sources=(
            DeviceLogSourceReceipt(
                source=DeviceLogSource(log_type="logcat", artifact_stem="logcat"),
                state=DeviceLogDrainState.NO_RETAINED_ITEMS,
                observed_item_count=2,
                filtered_item_count=2,
                diagnostic_hints=(
                    "Exception class: NumberFormatException",
                    "App frame: com.example.app.Editor.save(Editor.java:50)",
                ),
            ),
            DeviceLogSourceReceipt(
                source=DeviceLogSource(
                    log_type="exitinfo",
                    artifact_stem="exit-info",
                    is_diagnostic=True,
                ),
                state=DeviceLogDrainState.CAPTURED,
                observed_item_count=1,
                retained_item_count=1,
                artifacts=(artifact,),
                observed_failure_kinds=(ApplicationFailureKind.CRASH,),
                diagnostic_classification=(ApplicationDiagnosticClassification.APPLICATION_CRASH),
            ),
        ),
    )

    failure = action_application_failure(receipt)

    assert failure is not None
    assert failure.classification is ApplicationDiagnosticClassification.APPLICATION_CRASH
    assert failure.source_artifact_ids == ("diagnostic_1",)
    assert failure.diagnostic_hints == (
        "Exception class: NumberFormatException",
        "App frame: com.example.app.Editor.save(Editor.java:50)",
    )
    assert failure.summary == (
        "Platform diagnostics established that the application crashed after this action. "
        "Platform evidence: APP CRASH(EXCEPTION) reason=4,status=0."
    )
