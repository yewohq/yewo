"""Canonical run evidence consumed by every report renderer."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yewo.artifacts import (
    RelativeArtifactPath,
    CANONICAL_RUN_RECORD_SCHEMA_VERSION as CANONICAL_RUN_RECORD_SCHEMA_VERSION,
    CanonicalRunRecordSchemaVersion,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
)
from yewo.runs.records import RunMode, RunStage, RunTerminalCategory
from yewo.reporting.audit_records import (
    AuditFindingRecord as AuditFindingRecord,
    AuditFindingRegionRecord as AuditFindingRegionRecord,
    AuditFindingVerificationKind as AuditFindingVerificationKind,
    AuditObligationRecord as AuditObligationRecord,
    AuditObligationStatus as AuditObligationStatus,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
)
from yewo.reporting.limits import REPORT_REASON_MAX_LENGTH, REPORT_TEXT_MAX_LENGTH


class ScenarioExecutionClassification(StrEnum):
    """Why scenario execution stopped."""

    COMPLETED = "completed"
    INSUFFICIENT_DISCOVERY = "insufficient_discovery"
    UNSUPPORTED_APPLICATION_BEHAVIOR = "unsupported_application_behavior"
    APPLICATION_FAILURE = "application_failure"
    APPLICATION_CRASH = "application_crash"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CAPABILITY_FAILURE = "capability_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CanonicalRunCompleteness(StrEnum):
    """Whether Yewo completed all requested scenario work."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


class InteractionGeometrySource(StrEnum):
    """Evidence source that supplied one interaction location."""

    MODEL = "model"
    SCREEN_ANALYSIS = "screen_analysis"
    MEMORY = "memory"
    PROCEDURE = "procedure"
    AUTHORIZATION = "authorization"


class ReportOutputFormat(StrEnum):
    """One human or CI report derived from the canonical record."""

    MARKDOWN = "markdown"
    HTML = "html"
    JUNIT = "junit"


class ReportOutputDisposition(StrEnum):
    """Whether one report output was written."""

    WRITTEN = "written"
    FAILED = "failed"


class CanonicalEvidenceKind(StrEnum):
    """Report-facing purpose of one persisted artifact."""

    SCREENSHOT = "screenshot"
    RECORDING = "recording"
    DEVICE_LOG = "device_log"
    MODEL_CALL_LEDGER = "model_call_ledger"
    EXECUTION_TRACE = "execution_trace"
    CAPABILITY_RECEIPT = "capability_receipt"
    LIFECYCLE_RECEIPT = "lifecycle_receipt"
    RESTORATION_RECEIPT = "restoration_receipt"
    RECOVERY_RECEIPT = "recovery_receipt"
    DISCOVERY_SUMMARY = "discovery_summary"
    ROUTE_SUMMARY = "route_summary"
    SCENARIO_AUDIT = "scenario_audit"
    USAGE = "usage"


class CanonicalEvidenceReference(BaseModel):
    """Safe report link to one durable artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_id: RecordIdentifier
    kind: CanonicalEvidenceKind
    path: RelativeArtifactPath
    label: str = Field(min_length=1, max_length=200)
    scenario_id: RecordIdentifier | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts or value.name in {"", "."}:
            raise ValueError("A report evidence path must be a safe relative path.")
        return value


class CanonicalUsageSummary(BaseModel):
    """Known model usage without treating missing provider usage as zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_call_count: int = Field(default=0, ge=0)
    calls_with_unknown_usage: int = Field(default=0, ge=0)
    known_prompt_tokens: int = Field(default=0, ge=0)
    known_completion_tokens: int = Field(default=0, ge=0)
    known_cached_tokens: int = Field(default=0, ge=0)
    known_cost_usd: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def _validate_unknown_usage_count(self) -> Self:
        if self.calls_with_unknown_usage > self.model_call_count:
            raise ValueError("Unknown usage calls cannot exceed the model call count.")
        return self


class CanonicalPoint(BaseModel):
    """One exact point in source-screenshot pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class CanonicalRegion(BaseModel):
    """One half-open region in source-screenshot pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_positive_area(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("An interaction region must have positive area.")
        return self


class CanonicalInteractionRecord(BaseModel):
    """Report-facing evidence for one dispatched pointer action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    interaction_id: RecordIdentifier
    action: str = Field(min_length=1, max_length=100)
    source_observation_id: RecordIdentifier
    screenshot_evidence_id: RecordIdentifier
    screenshot_width_px: int = Field(gt=0)
    screenshot_height_px: int = Field(gt=0)
    point: CanonicalPoint
    region: CanonicalRegion | None = None
    geometry_source: InteractionGeometrySource
    target: str | None = Field(default=None, min_length=1, max_length=1_000)
    expected_outcome: str = Field(min_length=1, max_length=2_000)
    tap_count: int = Field(default=1, ge=1, le=10)
    dispatch_disposition: str = Field(min_length=1, max_length=100)
    outcome_disposition: str | None = Field(default=None, min_length=1, max_length=100)
    settled_observation_id: RecordIdentifier | None = None

    @model_validator(mode="after")
    def _validate_geometry_bounds(self) -> Self:
        if self.point.x >= self.screenshot_width_px or self.point.y >= self.screenshot_height_px:
            raise ValueError("An interaction point must be within its source screenshot.")
        if self.region is not None and (
            self.region.right > self.screenshot_width_px
            or self.region.bottom > self.screenshot_height_px
        ):
            raise ValueError("An interaction region must be within its source screenshot.")
        return self


class CanonicalModelCallSummary(BaseModel):
    """Safe report projection of one persisted provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: RecordIdentifier
    profile_id: RecordIdentifier
    role: str = Field(min_length=1, max_length=100)
    requested_provider: str = Field(min_length=1, max_length=200)
    requested_model: str = Field(min_length=1, max_length=500)
    served_model: str | None = Field(default=None, min_length=1, max_length=500)
    disposition: str = Field(min_length=1, max_length=100)
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    scenario_id: RecordIdentifier | None = None


class CanonicalReportOutput(BaseModel):
    """Observed result of writing one derived report."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    format: ReportOutputFormat
    disposition: ReportOutputDisposition
    path: RelativeArtifactPath
    reason: str | None = Field(default=None, min_length=1, max_length=1_024)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts or value.name in {"", "."}:
            raise ValueError("A report output path must be a safe relative path.")
        return value

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        if (self.disposition is ReportOutputDisposition.FAILED) != (self.reason is not None):
            raise ValueError("Only a failed report output requires a reason.")
        return self


class ScenarioExecutionRecord(BaseModel):
    """Final execution classification and its direct evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    classification: ScenarioExecutionClassification
    reason: str = Field(min_length=1)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    action_count: int = Field(default=0, ge=0)
    model_call_count: int = Field(default=0, ge=0)
    evidence_ids: tuple[RecordIdentifier, ...] = ()
    restoration_failure_reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=REPORT_REASON_MAX_LENGTH,
    )

    @model_validator(mode="after")
    def _validate_times(self) -> Self:
        if (self.started_at is None) is not (self.completed_at is None):
            raise ValueError("Scenario execution times must be both present or both absent.")
        for value in (self.started_at, self.completed_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("Scenario execution times must include a timezone.")
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("Scenario execution cannot finish before it starts.")
        return self


class CanonicalScenarioRecord(BaseModel):
    """One generated scenario, its execution, and its independent audit."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    scenario_id: RecordIdentifier
    journey_id: RecordIdentifier
    coverage_cell_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    name: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=REPORT_TEXT_MAX_LENGTH)
    expected_outcomes: tuple[
        Annotated[str, Field(min_length=1, max_length=REPORT_TEXT_MAX_LENGTH)], ...
    ] = Field(min_length=1)
    execution: ScenarioExecutionRecord
    audit: ScenarioAuditRecord
    interactions: tuple[CanonicalInteractionRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_execution_verdict(self) -> Self:
        if sorted(self.expected_outcomes) != sorted(
            obligation.description for obligation in self.audit.obligations
        ):
            raise ValueError("The scenario audit must settle every expected outcome once.")
        required_verdict = {
            ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY: (
                ScenarioAuditVerdict.UNVERIFIED
            ),
            ScenarioExecutionClassification.UNSUPPORTED_APPLICATION_BEHAVIOR: (
                ScenarioAuditVerdict.UNVERIFIED
            ),
            ScenarioExecutionClassification.BLOCKED_BY_POLICY: ScenarioAuditVerdict.UNVERIFIED,
            ScenarioExecutionClassification.UNSUPPORTED_CAPABILITY: (
                ScenarioAuditVerdict.UNVERIFIED
            ),
            ScenarioExecutionClassification.CAPABILITY_FAILURE: ScenarioAuditVerdict.ERROR,
            ScenarioExecutionClassification.BUDGET_EXHAUSTED: ScenarioAuditVerdict.UNVERIFIED,
            ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE: ScenarioAuditVerdict.ERROR,
            ScenarioExecutionClassification.INTERRUPTED: ScenarioAuditVerdict.ABORTED,
        }.get(self.execution.classification)
        if required_verdict is not None and self.audit.verdict is not required_verdict:
            raise ValueError(
                "The scenario audit verdict must agree with the execution classification."
            )
        return self


def _aggregate_qa_verdict(verdicts: Collection[object]) -> ScenarioAuditVerdict:
    for verdict in (
        ScenarioAuditVerdict.FAILED,
        ScenarioAuditVerdict.ERROR,
        ScenarioAuditVerdict.ABORTED,
        ScenarioAuditVerdict.UNVERIFIED,
    ):
        if verdict in verdicts:
            return verdict
    if verdicts == {ScenarioAuditVerdict.PASSED}:
        return ScenarioAuditVerdict.PASSED
    return ScenarioAuditVerdict.UNVERIFIED


class CanonicalRunRecord(BaseModel):
    """Versioned semantic source for JSON, HTML, Markdown, and JUnit."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: CanonicalRunRecordSchemaVersion = CANONICAL_RUN_RECORD_SCHEMA_VERSION
    run_id: RecordIdentifier
    mode: RunMode
    started_at: datetime
    completed_at: datetime
    terminal_category: RunTerminalCategory
    terminal_stage: RunStage
    terminal_reason: str = Field(min_length=1)
    partial_evidence_available: bool
    device: EvidenceDeviceIdentity | None = None
    application: EvidenceApplicationIdentity | None = None
    scenarios: tuple[CanonicalScenarioRecord, ...] = ()
    evidence: tuple[CanonicalEvidenceReference, ...] = ()
    usage: CanonicalUsageSummary = CanonicalUsageSummary()
    model_calls: tuple[CanonicalModelCallSummary, ...] = ()
    report_outputs: tuple[CanonicalReportOutput, ...] = ()
    qa_verdict: ScenarioAuditVerdict = ScenarioAuditVerdict.UNVERIFIED
    completeness: CanonicalRunCompleteness = CanonicalRunCompleteness.NONE

    @model_validator(mode="before")
    @classmethod
    def _populate_derived_statuses(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        scenarios = payload.get("scenarios") or ()
        if "qa_verdict" not in payload:
            verdicts = {
                (
                    scenario.audit.verdict
                    if isinstance(scenario, CanonicalScenarioRecord)
                    else scenario["audit"]["verdict"]
                )
                for scenario in scenarios
            }
            payload["qa_verdict"] = _aggregate_qa_verdict(verdicts)
        if "completeness" not in payload:
            terminal = payload.get("terminal_category")
            all_completed = all(
                (
                    scenario.execution.classification
                    if isinstance(scenario, CanonicalScenarioRecord)
                    else scenario["execution"]["classification"]
                )
                == ScenarioExecutionClassification.COMPLETED
                for scenario in scenarios
            )
            if terminal == RunTerminalCategory.COMPLETED and all_completed:
                payload["completeness"] = CanonicalRunCompleteness.COMPLETE
            elif payload.get("partial_evidence_available") or scenarios or payload.get("evidence"):
                payload["completeness"] = CanonicalRunCompleteness.PARTIAL
            else:
                payload["completeness"] = CanonicalRunCompleteness.NONE
        return payload

    def _derived_qa_verdict(self) -> ScenarioAuditVerdict:
        verdicts = {scenario.audit.verdict for scenario in self.scenarios}
        return _aggregate_qa_verdict(set(verdicts))

    def _derived_completeness(self) -> CanonicalRunCompleteness:
        if self.terminal_category is RunTerminalCategory.COMPLETED and all(
            scenario.execution.classification is ScenarioExecutionClassification.COMPLETED
            for scenario in self.scenarios
        ):
            return CanonicalRunCompleteness.COMPLETE
        if self.partial_evidence_available or self.scenarios or self.evidence:
            return CanonicalRunCompleteness.PARTIAL
        return CanonicalRunCompleteness.NONE

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        self._validate_times()
        self._validate_terminal()
        self._validate_scenario_ids()
        self._validate_evidence_references()
        self._validate_derived_statuses()
        return self

    def _validate_derived_statuses(self) -> None:
        qa_verdict = self._derived_qa_verdict()
        completeness = self._derived_completeness()
        if self.qa_verdict is not qa_verdict:
            raise ValueError("The canonical QA verdict does not match the scenario audits.")
        if self.completeness is not completeness:
            raise ValueError("Canonical completeness does not match the run evidence.")

    def _validate_times(self) -> None:
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Canonical run times must include a timezone.")
        if self.completed_at < self.started_at:
            raise ValueError("A canonical run cannot finish before it starts.")

    def _validate_terminal(self) -> None:
        if (
            self.terminal_category is RunTerminalCategory.COMPLETED
            and self.terminal_stage is not RunStage.FINISHED
        ):
            raise ValueError("A completed canonical run must finish every stage.")
        if (
            self.mode is RunMode.SCENARIO
            and self.terminal_category is RunTerminalCategory.COMPLETED
            and not self.scenarios
        ):
            raise ValueError("A completed scenario run must contain an executable scenario.")

    def _validate_scenario_ids(self) -> None:
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Canonical scenario identifiers must be unique.")

    def _validate_evidence_references(self) -> None:
        evidence_by_id = {reference.artifact_id: reference for reference in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("Canonical evidence identifiers must be unique.")
        scenario_ids = {scenario.scenario_id for scenario in self.scenarios}
        for reference in self.evidence:
            if reference.scenario_id is not None and reference.scenario_id not in scenario_ids:
                raise ValueError("Canonical evidence must reference a known scenario.")
        for scenario in self.scenarios:
            referenced_ids = set(scenario.execution.evidence_ids)
            referenced_ids.update(
                interaction.screenshot_evidence_id for interaction in scenario.interactions
            )
            for obligation in scenario.audit.obligations:
                referenced_ids.update(obligation.evidence_ids)
                referenced_ids.update(
                    region.screenshot_evidence_id
                    for finding in obligation.findings
                    for region in finding.regions
                    if region.screenshot_evidence_id is not None
                )
            if not referenced_ids.issubset(evidence_by_id):
                raise ValueError("A scenario references evidence outside the canonical registry.")
            if any(
                evidence_by_id[evidence_id].scenario_id not in {None, scenario.scenario_id}
                for evidence_id in referenced_ids
            ):
                raise ValueError("A scenario cannot reference evidence owned by another scenario.")
