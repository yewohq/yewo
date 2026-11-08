"""Build the canonical record produced by scenario execution and audit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from yewo.agent.discovery.conclusion import DISCOVERY_RESULT_PATH
from yewo.artifacts import ARTIFACTS_DIRECTORY, RUN_RECORDS_DIRECTORY
from yewo.reporting import (
    CanonicalEvidenceKind,
    CanonicalEvidenceReference,
    CanonicalInteractionRecord,
    CanonicalRunRecord,
    CanonicalRunRecordStore,
    CanonicalScenarioRecord,
    ScenarioAuditRecord,
    ReportBundleWriteError,
    ReportOutputDisposition,
)
from yewo.reporting.bundle import write_report_bundle
from yewo.reporting.evidence import (
    build_audit_finding_evidence,
    build_interaction_evidence,
    load_model_call_evidence,
)
from yewo.runs import RunStart, RunTerminal
from yewo.scenarios.coordinator import ScenarioCoordinationResult
from yewo.scenarios.auditor import (
    scenario_audit_artifact_id,
    scenario_execution_artifact_id,
)
from yewo.scenarios.records import ScenarioDefinition, ScenarioGenerationResult
from yewo.scenarios.paths import (
    SCENARIO_GENERATION_PATH,
    scenario_artifact_directory,
    scenario_audit_path,
    scenario_execution_path,
)
from yewo.scenarios.workflow import ScenarioTestWorkflow


def persist_run_record(
    run_root: Path,
    terminal: RunTerminal,
    workflow: ScenarioTestWorkflow,
) -> Path:
    """Persist execution evidence and independent scenario verdicts."""
    start = _load_start(run_root)
    generation = workflow.generation
    interaction_records = {
        item.scenario_id: build_interaction_evidence(item.scenario_id, item)
        for item in workflow.executions
    }
    audit_records: dict[str, ScenarioAuditRecord] = {}
    audit_finding_evidence: list[CanonicalEvidenceReference] = []
    for item in workflow.executions:
        audit, references = build_audit_finding_evidence(
            run_root / scenario_artifact_directory(item.scenario_id),
            item.scenario_id,
            workflow.audits[item.scenario_id],
        )
        audit_records[item.scenario_id] = audit
        audit_finding_evidence.extend(references)
    scenarios = (
        _canonical_scenarios(
            generation,
            workflow.executions,
            audit_records,
            {key: value[0] for key, value in interaction_records.items()},
        )
        if generation is not None
        else ()
    )
    scenario_evidence = tuple(
        reference
        for item in workflow.executions
        for reference in (
            CanonicalEvidenceReference(
                artifact_id=scenario_execution_artifact_id(item.scenario_id),
                kind=CanonicalEvidenceKind.EXECUTION_TRACE,
                path=scenario_execution_path(item.scenario_id),
                label="Scenario execution evidence",
                scenario_id=item.scenario_id,
            ),
            CanonicalEvidenceReference(
                artifact_id=scenario_audit_artifact_id(item.scenario_id),
                kind=CanonicalEvidenceKind.SCENARIO_AUDIT,
                path=scenario_audit_path(item.scenario_id),
                label="Independent scenario audit",
                scenario_id=item.scenario_id,
            ),
            *_execution_detail_evidence(item),
        )
        if item.scenario_id in workflow.audits
    )
    interaction_evidence = tuple(
        reference
        for unused_records, references in interaction_records.values()
        for reference in references
    )
    model_calls, usage, model_evidence = load_model_call_evidence(
        run_root,
        tuple(item.scenario_id for item in workflow.executions),
    )
    evidence = (
        *scenario_evidence,
        *_workflow_evidence(run_root, terminal.run_id, workflow),
        *interaction_evidence,
        *audit_finding_evidence,
        *model_evidence,
    )
    record = CanonicalRunRecord(
        run_id=terminal.run_id,
        mode=start.request.mode,
        started_at=start.started_at,
        completed_at=terminal.completed_at,
        terminal_category=terminal.category,
        terminal_stage=terminal.stage,
        terminal_reason=terminal.reason,
        partial_evidence_available=terminal.partial_evidence_available,
        device=workflow.device_identity,
        application=(generation.suite.application if generation is not None else None),
        scenarios=scenarios,
        evidence=evidence,
        usage=usage,
        model_calls=model_calls,
    )
    report_outputs = write_report_bundle(run_root, record)
    record = record.model_copy(update={"report_outputs": report_outputs})
    path = CanonicalRunRecordStore(run_root).persist(record)
    if any(output.disposition is ReportOutputDisposition.FAILED for output in report_outputs):
        raise ReportBundleWriteError(
            "One or more reports could not be written. See reports/run.json."
        )
    return path


def _workflow_evidence(
    run_root: Path,
    run_id: str,
    workflow: ScenarioTestWorkflow,
) -> tuple[CanonicalEvidenceReference, ...]:
    references: list[CanonicalEvidenceReference] = []
    fixed = (
        (
            "discovery",
            CanonicalEvidenceKind.DISCOVERY_SUMMARY,
            DISCOVERY_RESULT_PATH,
            "Discovery result",
        ),
        (
            "generation",
            CanonicalEvidenceKind.ROUTE_SUMMARY,
            SCENARIO_GENERATION_PATH,
            "Generated scenario suite and route evidence",
        ),
    )
    for prefix, kind, path, label in fixed:
        if (run_root / path).is_file():
            references.append(
                CanonicalEvidenceReference(
                    artifact_id=_report_artifact_id(prefix, run_id),
                    kind=kind,
                    path=path,
                    label=label,
                )
            )
    for recording_path in sorted((run_root / ARTIFACTS_DIRECTORY / "recordings").glob("**/*.mp4")):
        if recording_path.is_file() and not recording_path.is_symlink():
            relative_path = recording_path.relative_to(run_root)
            references.append(
                CanonicalEvidenceReference(
                    artifact_id=_report_artifact_id("recording", relative_path.as_posix()),
                    kind=CanonicalEvidenceKind.RECORDING,
                    path=relative_path,
                    label="Run screen recording",
                )
            )
    for result in workflow.executions:
        execution_path = scenario_execution_path(result.scenario_id)
        if result.route is not None:
            references.append(
                CanonicalEvidenceReference(
                    artifact_id=_report_artifact_id("route", result.scenario_id),
                    kind=CanonicalEvidenceKind.ROUTE_SUMMARY,
                    path=execution_path,
                    label="Scenario route replay evidence",
                    scenario_id=result.scenario_id,
                )
            )
        for receipt in result.capability_receipts:
            references.extend(
                (
                    CanonicalEvidenceReference(
                        artifact_id=_report_artifact_id(
                            "capability", result.scenario_id, receipt.transaction_id
                        ),
                        kind=CanonicalEvidenceKind.LIFECYCLE_RECEIPT,
                        path=execution_path,
                        label=f"{receipt.request.kind.value} capability receipt",
                        scenario_id=result.scenario_id,
                    ),
                    CanonicalEvidenceReference(
                        artifact_id=_report_artifact_id(
                            "restoration", result.scenario_id, receipt.transaction_id
                        ),
                        kind=CanonicalEvidenceKind.RESTORATION_RECEIPT,
                        path=execution_path,
                        label=f"{receipt.request.kind.value} restoration receipt",
                        scenario_id=result.scenario_id,
                    ),
                )
            )
    return tuple(references)


def _execution_detail_evidence(
    result: ScenarioCoordinationResult,
) -> tuple[CanonicalEvidenceReference, ...]:
    execution_path = scenario_execution_path(result.scenario_id)
    return tuple(
        CanonicalEvidenceReference(
            artifact_id=evidence_id,
            kind=CanonicalEvidenceKind.EXECUTION_TRACE,
            path=execution_path,
            label="Scenario execution evidence item",
            scenario_id=result.scenario_id,
        )
        for evidence_id in result.execution.evidence_ids
    )


def _report_artifact_id(prefix: str, *parts: str) -> str:
    digest = sha256(":".join((prefix, *parts)).encode()).hexdigest()[:48]
    return f"{prefix}_{digest}"


def _load_start(run_root: Path) -> RunStart:
    path = run_root / RUN_RECORDS_DIRECTORY / "start.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("The test run start record is unavailable.")
    try:
        return RunStart.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RuntimeError("Could not read the test run start record.") from error


def _canonical_scenarios(
    generation: ScenarioGenerationResult,
    executions: tuple[ScenarioCoordinationResult, ...],
    audits: dict[str, ScenarioAuditRecord],
    interactions: dict[str, tuple[CanonicalInteractionRecord, ...]] | None = None,
) -> tuple[CanonicalScenarioRecord, ...]:
    definitions = {item.scenario_id: item for item in generation.suite.scenarios}
    return tuple(
        _canonical_scenario(
            definitions[item.scenario_id],
            item,
            audits[item.scenario_id],
            (interactions or {}).get(item.scenario_id, ()),
        )
        for item in executions
    )


def _canonical_scenario(
    scenario: ScenarioDefinition,
    result: ScenarioCoordinationResult,
    audit: ScenarioAuditRecord,
    interactions: tuple[CanonicalInteractionRecord, ...] = (),
) -> CanonicalScenarioRecord:
    expected_outcomes = tuple(
        outcome for obligation in scenario.obligations for outcome in obligation.expected_outcomes
    )
    coverage_cell_id = next(
        (
            item.coverage_cell_id
            for item in scenario.obligations
            if item.coverage_cell_id is not None
        ),
        (f"{scenario.obligations[0].category.value}.{scenario.obligations[0].strategy.value}.0"),
    )
    return CanonicalScenarioRecord(
        scenario_id=scenario.scenario_id,
        journey_id=scenario.journey_id,
        coverage_cell_id=coverage_cell_id,
        name=scenario.name,
        objective=scenario.objective,
        expected_outcomes=tuple(item.statement for item in expected_outcomes),
        execution=result.execution,
        audit=audit,
        interactions=interactions,
    )
