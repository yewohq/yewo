"""Tests for immutable device-log evidence records."""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
import pytest

from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
)
from yewo.devices import DeviceKind, DevicePlatform
from yewo.devices.logs import (
    DeviceLogArtifactKind,
    DeviceLogArtifactReference,
    DeviceLogBoundary,
    DeviceLogBoundarySource,
    DeviceLogBoundaryState,
    DeviceLogCheckpointKind,
    DeviceLogDrainReceipt,
    DeviceLogDrainState,
    DeviceLogOmission,
    DeviceLogSource,
    DeviceLogSourceReceipt,
)
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)


CAPTURE_TIME = datetime(2026, 10, 30, tzinfo=timezone.utc)
SOURCE = DeviceLogSource(log_type="logcat", artifact_stem="logcat")
DEVICE = EvidenceDeviceIdentity(
    target_id="emulator-5554",
    platform=DevicePlatform.ANDROID,
    kind=DeviceKind.VIRTUAL,
)
APPLICATION = EvidenceApplicationIdentity(
    platform=DevicePlatform.ANDROID,
    application_id="com.example.app",
)


def _artifact(*, boundary_id: str = "boundary_1") -> DeviceLogArtifactReference:
    return DeviceLogArtifactReference(
        artifact_id="artifact_1",
        boundary_id=boundary_id,
        drain_id="drain_1",
        log_type="logcat",
        kind=DeviceLogArtifactKind.TEXT_STREAM,
        sha256="0" * 64,
        path=Path(f"device-logs/{boundary_id}/log.log"),
        byte_count=1,
        entry_count=1,
        redaction_policy_id="test-policy",
        source_redaction_count=0,
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def _receipt(**updates: object) -> DeviceLogDrainReceipt:
    values: dict[str, object] = {
        "drain_id": "drain_1",
        "boundary_id": "boundary_1",
        "drain_index": 1,
        "checkpoint_kind": DeviceLogCheckpointKind.ACTION,
        "checkpoint_id": "action_1",
        "started_at": CAPTURE_TIME,
        "completed_at": CAPTURE_TIME,
        "sources": (
            DeviceLogSourceReceipt(
                source=SOURCE,
                state=DeviceLogDrainState.NO_RETAINED_ITEMS,
            ),
        ),
    }
    values.update(updates)
    return DeviceLogDrainReceipt.model_validate(values)


def test_boundary_rejects_naive_time() -> None:
    with pytest.raises(ValidationError, match="include a timezone"):
        DeviceLogBoundary(
            boundary_id="boundary_1",
            device=DEVICE,
            application=APPLICATION,
            started_at=datetime(2026, 10, 30),
            sources=(
                DeviceLogBoundarySource(
                    source=SOURCE,
                    state=DeviceLogBoundaryState.READY,
                    baseline_item_count=0,
                    may_have_omitted_earlier_items=False,
                ),
            ),
        )


def test_boundary_source_requires_truthful_reason() -> None:
    with pytest.raises(ValidationError, match="must have a reason"):
        DeviceLogBoundarySource(
            source=SOURCE,
            state=DeviceLogBoundaryState.UNSUPPORTED,
            baseline_item_count=0,
            may_have_omitted_earlier_items=False,
        )

    with pytest.raises(ValidationError, match="cannot have a failure reason"):
        DeviceLogBoundarySource(
            source=SOURCE,
            state=DeviceLogBoundaryState.READY,
            baseline_item_count=0,
            may_have_omitted_earlier_items=False,
            reason="not ready",
        )


def test_source_accepts_vendor_spelling_but_keeps_a_safe_artifact_stem() -> None:
    source = DeviceLogSource(
        log_type="safariConsole",
        artifact_stem="safari-console",
    )

    assert source.log_type == "safariConsole"


def test_diagnostic_classification_survives_omitted_artifact_bytes() -> None:
    receipt = DeviceLogSourceReceipt(
        source=DeviceLogSource(
            log_type="exitinfo",
            artifact_stem="exit-info",
            is_diagnostic=True,
        ),
        state=DeviceLogDrainState.NO_RETAINED_ITEMS,
        observed_item_count=1,
        omission=DeviceLogOmission.LATER,
        observed_failure_kinds=(ApplicationFailureKind.ANR,),
        diagnostic_classification=ApplicationDiagnosticClassification.APPLICATION_ANR,
    )

    assert receipt.artifacts == ()


def test_unparsed_diagnostics_cannot_be_reported_as_no_failure() -> None:
    receipt = DeviceLogSourceReceipt(
        source=DeviceLogSource(
            log_type="crashlog",
            artifact_stem="crashlog",
            is_diagnostic=True,
        ),
        state=DeviceLogDrainState.NO_RETAINED_ITEMS,
        observed_item_count=1,
        unparsed_item_count=1,
        diagnostic_classification=ApplicationDiagnosticClassification.DIAGNOSTICS_INCOMPLETE,
    )

    assert receipt.unparsed_item_count == 1


def test_source_rejects_overlapping_filtered_and_unparsed_counts() -> None:
    with pytest.raises(ValidationError, match="exclude more items"):
        DeviceLogSourceReceipt(
            source=SOURCE,
            state=DeviceLogDrainState.NO_RETAINED_ITEMS,
            observed_item_count=1,
            filtered_item_count=1,
            unparsed_item_count=1,
        )


def test_drain_rejects_an_artifact_from_another_boundary() -> None:
    with pytest.raises(ValidationError, match="must match its boundary"):
        _receipt(
            sources=(
                DeviceLogSourceReceipt(
                    source=SOURCE,
                    state=DeviceLogDrainState.CAPTURED,
                    observed_item_count=1,
                    retained_item_count=1,
                    artifacts=(_artifact(boundary_id="other_boundary"),),
                ),
            )
        )


def test_drain_rejects_repeated_sources() -> None:
    empty_receipt = DeviceLogSourceReceipt(
        source=SOURCE,
        state=DeviceLogDrainState.NO_RETAINED_ITEMS,
    )

    with pytest.raises(ValidationError, match="cannot repeat a source"):
        _receipt(sources=(empty_receipt, empty_receipt))


@pytest.mark.parametrize(
    ("checkpoint_kind", "checkpoint_id"),
    [
        (DeviceLogCheckpointKind.ACTION, None),
        (DeviceLogCheckpointKind.APPLICATION_SETUP, None),
        (DeviceLogCheckpointKind.SCENARIO, None),
        (DeviceLogCheckpointKind.SESSION_END, "scenario_1"),
        (DeviceLogCheckpointKind.SESSION_START, "action_1"),
    ],
)
def test_drain_rejects_checkpoint_identity_mismatch(
    checkpoint_kind: DeviceLogCheckpointKind,
    checkpoint_id: str | None,
) -> None:
    with pytest.raises(ValidationError, match="checkpoint ID"):
        _receipt(checkpoint_kind=checkpoint_kind, checkpoint_id=checkpoint_id)


def test_drain_rejects_completion_before_start() -> None:
    with pytest.raises(ValidationError, match="before it starts"):
        _receipt(
            started_at=datetime(2026, 10, 30, 1, tzinfo=timezone.utc),
            completed_at=CAPTURE_TIME,
        )
