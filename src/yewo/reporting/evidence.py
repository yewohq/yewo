"""Project persisted execution evidence into the canonical report contract."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path

from yewo.agent.runtime.records import AgentStep
from yewo.artifacts import ARTIFACTS_DIRECTORY, RecordIdentifier
from yewo.grounding import GroundedRegion, NormalizedRegion
from yewo.observations import ObservationCaptureError, ObservationStore
from yewo.reporting.records import (
    CanonicalEvidenceKind,
    CanonicalEvidenceReference,
    CanonicalInteractionRecord,
    CanonicalModelCallSummary,
    CanonicalPoint,
    CanonicalRegion,
    CanonicalUsageSummary,
    InteractionGeometrySource,
    ScenarioAuditRecord,
)
from yewo.scenarios.coordinator import ScenarioCoordinationResult
from yewo.scenarios.paths import scenario_artifact_directory, scenario_child_artifact_path


def build_audit_finding_evidence(
    child_root: Path,
    scenario_id: RecordIdentifier,
    audit: ScenarioAuditRecord,
) -> tuple[ScenarioAuditRecord, tuple[CanonicalEvidenceReference, ...]]:
    """Link confirmed findings to their exact current-run screenshots."""
    observation_ids = {
        region.observation_id
        for obligation in audit.obligations
        for finding in obligation.findings
        for region in finding.regions
    }
    if not observation_ids:
        return audit, ()
    try:
        observations = ObservationStore(child_root).load_selected(observation_ids)
    except ObservationCaptureError:
        return audit, ()
    observation_by_id = {item.observation_id: item for item in observations}
    evidence_by_id: dict[RecordIdentifier, CanonicalEvidenceReference] = {}
    obligations = []
    for obligation in audit.obligations:
        finding_evidence_ids: list[RecordIdentifier] = []
        findings = []
        for finding in obligation.findings:
            regions = []
            for region in finding.regions:
                observation = observation_by_id.get(region.observation_id)
                if observation is None:
                    regions.append(region)
                    continue
                screenshot = observation.screenshot
                evidence_id = _derived_id(
                    "audit-screenshot",
                    scenario_id,
                    screenshot.artifact_id,
                )
                evidence_by_id[evidence_id] = CanonicalEvidenceReference(
                    artifact_id=evidence_id,
                    kind=CanonicalEvidenceKind.SCREENSHOT,
                    path=scenario_child_artifact_path(scenario_id, screenshot.path),
                    label=region.label,
                    scenario_id=scenario_id,
                )
                finding_evidence_ids.append(evidence_id)
                regions.append(region.model_copy(update={"screenshot_evidence_id": evidence_id}))
            findings.append(finding.model_copy(update={"regions": tuple(regions)}))
        obligations.append(
            obligation.model_copy(
                update={
                    "evidence_ids": tuple(
                        dict.fromkeys((*obligation.evidence_ids, *finding_evidence_ids))
                    ),
                    "findings": tuple(findings),
                }
            )
        )
    return (
        audit.model_copy(update={"obligations": tuple(obligations)}),
        tuple(evidence_by_id.values()),
    )


def build_interaction_evidence(
    scenario_id: RecordIdentifier,
    result: ScenarioCoordinationResult,
) -> tuple[tuple[CanonicalInteractionRecord, ...], tuple[CanonicalEvidenceReference, ...]]:
    """Return report interactions and their source screenshots."""
    steps = (
        *(result.route.agent_result.steps if result.route is not None else ()),
        *(result.agent.steps if result.agent is not None else ()),
    )
    interactions: list[CanonicalInteractionRecord] = []
    screenshots: dict[str, CanonicalEvidenceReference] = {}
    for step in steps:
        interaction = _interaction_of(scenario_id, step)
        if interaction is None or step.authorization is None:
            continue
        grounding = step.target_grounding
        if grounding is None:
            continue
        screenshot = grounding.request.screenshot
        evidence_id = interaction.screenshot_evidence_id
        screenshots[evidence_id] = CanonicalEvidenceReference(
            artifact_id=evidence_id,
            kind=CanonicalEvidenceKind.SCREENSHOT,
            path=scenario_child_artifact_path(scenario_id, screenshot.path),
            label=f"Before {step.authorization.action.value}",
            scenario_id=scenario_id,
        )
        interactions.append(interaction)
    return tuple(interactions), tuple(screenshots.values())


def _interaction_of(
    scenario_id: RecordIdentifier,
    step: AgentStep,
) -> CanonicalInteractionRecord | None:
    authorization = step.authorization
    grounding = step.target_grounding
    if authorization is None or authorization.target is None or grounding is None:
        return None
    screenshot = grounding.request.screenshot
    candidate = grounding.request.screen_analysis_candidate
    location = grounding.result.location
    if isinstance(location, GroundedRegion):
        region = CanonicalRegion(
            left=location.pixel.left,
            top=location.pixel.top,
            right=location.pixel.right,
            bottom=location.pixel.bottom,
        )
    elif candidate is not None:
        region = _pixel_region(candidate.region, screenshot.width_px, screenshot.height_px)
    else:
        region = None
    target = step.proposal.target_expression
    if target is None and step.proposal.target_element_ref is not None:
        target = f"screen element {step.proposal.target_element_ref}"
    evidence_id = _derived_id("interaction-screenshot", scenario_id, screenshot.artifact_id)
    return CanonicalInteractionRecord(
        interaction_id=step.step_id,
        action=authorization.action.value,
        source_observation_id=step.source_observation_id,
        screenshot_evidence_id=evidence_id,
        screenshot_width_px=screenshot.width_px,
        screenshot_height_px=screenshot.height_px,
        point=CanonicalPoint(x=authorization.target.x, y=authorization.target.y),
        region=region,
        geometry_source=InteractionGeometrySource(grounding.result.resolution.kind),
        target=target,
        expected_outcome=authorization.expected_outcome,
        tap_count=authorization.tap_count,
        dispatch_disposition=(
            step.receipt.disposition.value if step.receipt is not None else "unavailable"
        ),
        outcome_disposition=(step.outcome.disposition.value if step.outcome is not None else None),
        settled_observation_id=step.settled_observation_id,
    )


def _pixel_region(
    region: NormalizedRegion,
    width_px: int,
    height_px: int,
) -> CanonicalRegion:
    return CanonicalRegion(
        left=max(0, math.floor(region.left * width_px)),
        top=max(0, math.floor(region.top * height_px)),
        right=min(width_px, max(1, math.ceil(region.right * width_px))),
        bottom=min(height_px, max(1, math.ceil(region.bottom * height_px))),
    )


def load_model_call_evidence(
    run_root: Path,
    scenario_ids: tuple[RecordIdentifier, ...],
) -> tuple[
    tuple[CanonicalModelCallSummary, ...],
    CanonicalUsageSummary,
    tuple[CanonicalEvidenceReference, ...],
]:
    """Load safe call metadata from every run-owned call ledger."""
    ledgers: list[tuple[RecordIdentifier | None, Path]] = [
        (None, run_root / ARTIFACTS_DIRECTORY / "models" / "calls.jsonl")
    ]
    ledgers.extend(
        (
            scenario_id,
            run_root
            / scenario_artifact_directory(scenario_id)
            / ARTIFACTS_DIRECTORY
            / "models"
            / "calls.jsonl",
        )
        for scenario_id in scenario_ids
    )
    calls: list[CanonicalModelCallSummary] = []
    references: list[CanonicalEvidenceReference] = []
    for scenario_id, path in ledgers:
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(run_root)
        references.append(
            CanonicalEvidenceReference(
                artifact_id=_derived_id("model-call-ledger", scenario_id or "run"),
                kind=CanonicalEvidenceKind.MODEL_CALL_LEDGER,
                path=relative_path,
                label="Model-call ledger",
                scenario_id=scenario_id,
            )
        )
        calls.extend(_load_call_rows(path, scenario_id))
    return tuple(calls), _usage_of(calls), tuple(references)


def _load_call_rows(
    path: Path,
    scenario_id: RecordIdentifier | None,
) -> tuple[CanonicalModelCallSummary, ...]:
    rows: list[CanonicalModelCallSummary] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            payload = json.loads(line)
            usage = payload.get("usage") or {}
            rows.append(
                CanonicalModelCallSummary(
                    request_id=payload["request_id"],
                    profile_id=payload["profile_id"],
                    role=payload["role"],
                    requested_provider=payload["requested_provider"],
                    requested_model=payload["requested_model"],
                    served_model=payload.get("served_model"),
                    disposition=payload["disposition"],
                    latency_ms=payload.get("latency_ms"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cached_tokens=usage.get("cached_tokens"),
                    cost_usd=usage.get("cost_usd"),
                    scenario_id=scenario_id,
                )
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"The model-call ledger is invalid: {path.name}.") from error
    return tuple(rows)


def _usage_of(calls: list[CanonicalModelCallSummary]) -> CanonicalUsageSummary:
    return CanonicalUsageSummary(
        model_call_count=len(calls),
        calls_with_unknown_usage=sum(
            call.prompt_tokens is None or call.completion_tokens is None for call in calls
        ),
        known_prompt_tokens=sum(call.prompt_tokens or 0 for call in calls),
        known_completion_tokens=sum(call.completion_tokens or 0 for call in calls),
        known_cached_tokens=sum(call.cached_tokens or 0 for call in calls),
        known_cost_usd=sum(
            (call.cost_usd or Decimal(0) for call in calls),
            start=Decimal(0),
        ),
    )


def _derived_id(prefix: str, *parts: str) -> str:
    digest = sha256(":".join((prefix, *parts)).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
