"""Finding-to-screenshot projection for canonical reports."""

from io import BytesIO

from PIL import Image

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.observations import ObservationStore, RedactionProvenance, ScreenshotStore
from yewo.reporting import (
    AuditFindingRecord,
    AuditFindingRegionRecord,
    AuditFindingVerificationKind,
    AuditObligationRecord,
    AuditObligationStatus,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
)
from yewo.reporting.evidence import build_audit_finding_evidence
from yewo.scenarios.paths import scenario_child_artifact_path

from observations.factories import build_observation


def _png() -> bytes:
    image = Image.new("RGB", (4, 6), color=(20, 30, 40))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_confirmed_finding_registers_its_exact_screenshot(tmp_path) -> None:
    screenshot = ScreenshotStore(
        tmp_path,
        identifier_factory=lambda: "finding_screen",
    ).persist_redacted(
        _png(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
        redaction=RedactionProvenance(
            policy_id="test-redaction",
            region_count=0,
            region_set_sha256="0" * 64,
            input_sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        ),
    )
    observation = build_observation(redacted_screenshot=screenshot)
    ObservationStore(tmp_path).persist(observation)
    finding = AuditFindingRecord(
        finding_id="finding_1",
        claim="Required content is clipped.",
        reason="The auditor located a directly visible defect.",
        verification_kind=AuditFindingVerificationKind.VISUAL,
        regions=(
            AuditFindingRegionRecord(
                observation_id=observation.observation_id,
                label="Clipped content",
                left=100,
                top=100,
                right=900,
                bottom=900,
            ),
        ),
    )
    audit = ScenarioAuditRecord(
        verdict=ScenarioAuditVerdict.FAILED,
        summary="A visible defect was confirmed.",
        obligations=(
            AuditObligationRecord(
                obligation_id="outcome_1",
                description="Required content remains visible.",
                status=AuditObligationStatus.UNMET,
                evidence_ids=("execution_1",),
                reason="Required content is clipped.",
                findings=(finding,),
            ),
        ),
    )

    projected, references = build_audit_finding_evidence(
        tmp_path,
        "scenario_1",
        audit,
    )

    region = projected.obligations[0].findings[0].regions[0]
    assert region.screenshot_evidence_id is not None
    assert projected.obligations[0].evidence_ids == (
        "execution_1",
        region.screenshot_evidence_id,
    )
    assert len(references) == 1
    assert references[0].artifact_id == region.screenshot_evidence_id
    assert references[0].path == scenario_child_artifact_path("scenario_1", screenshot.path)
