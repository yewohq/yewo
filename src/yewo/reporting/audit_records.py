"""Canonical independent-audit records."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.reporting.limits import REPORT_REASON_MAX_LENGTH, REPORT_TEXT_MAX_LENGTH


class ScenarioAuditVerdict(StrEnum):
    """Independent verdict for one scenario."""

    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    ERROR = "error"
    ABORTED = "aborted"


class AuditObligationStatus(StrEnum):
    """Evidence state of one expected outcome or audit rule."""

    SATISFIED = "satisfied"
    UNMET = "unmet"
    UNVERIFIED = "unverified"


class AuditFindingVerificationKind(StrEnum):
    """How code confirmed one auditor finding."""

    ARITHMETIC = "arithmetic"
    EQUALITY = "equality"
    VISUAL = "visual"


class AuditFindingRegionRecord(BaseModel):
    """One observation region supplied as evidence for a confirmed finding."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation_id: RecordIdentifier
    label: str = Field(min_length=1, max_length=200)
    left: int = Field(ge=0, le=1_000)
    top: int = Field(ge=0, le=1_000)
    right: int = Field(ge=0, le=1_000)
    bottom: int = Field(ge=0, le=1_000)
    screenshot_evidence_id: RecordIdentifier | None = None

    @model_validator(mode="after")
    def _validate_positive_area(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("An audit finding region must have positive area.")
        return self


class AuditFindingRecord(BaseModel):
    """One confirmed issue and the observations that establish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    finding_id: RecordIdentifier
    claim: str = Field(min_length=1, max_length=REPORT_REASON_MAX_LENGTH)
    reason: str = Field(min_length=1, max_length=REPORT_REASON_MAX_LENGTH)
    verification_kind: AuditFindingVerificationKind
    regions: tuple[AuditFindingRegionRecord, ...] = Field(min_length=1, max_length=20)


class AuditObligationRecord(BaseModel):
    """One expected outcome or rule and the evidence that settles it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    obligation_id: RecordIdentifier
    description: str = Field(min_length=1, max_length=REPORT_TEXT_MAX_LENGTH)
    status: AuditObligationStatus
    evidence_ids: tuple[RecordIdentifier, ...] = ()
    reason: str = Field(min_length=1, max_length=REPORT_REASON_MAX_LENGTH)
    findings: tuple[AuditFindingRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_satisfied_evidence(self) -> Self:
        if self.status is AuditObligationStatus.SATISFIED and not self.evidence_ids:
            raise ValueError("A satisfied audit obligation requires evidence.")
        if self.status is not AuditObligationStatus.UNMET and self.findings:
            raise ValueError("Only an unmet audit obligation can contain confirmed findings.")
        return self


class ScenarioAuditRecord(BaseModel):
    """Independent scenario verdict derived from persisted obligations."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    verdict: ScenarioAuditVerdict
    summary: str = Field(min_length=1, max_length=REPORT_REASON_MAX_LENGTH)
    obligations: tuple[AuditObligationRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_verdict_obligations(self) -> Self:
        statuses = {obligation.status for obligation in self.obligations}
        if self.verdict is ScenarioAuditVerdict.PASSED and statuses != {
            AuditObligationStatus.SATISFIED
        }:
            raise ValueError("A passed scenario requires every audit obligation to be satisfied.")
        if (
            self.verdict is ScenarioAuditVerdict.FAILED
            and AuditObligationStatus.UNMET not in statuses
        ):
            raise ValueError("A failed scenario requires an unmet audit obligation.")
        if (
            self.verdict is ScenarioAuditVerdict.UNVERIFIED
            and AuditObligationStatus.UNVERIFIED not in statuses
        ):
            raise ValueError("An unverified scenario requires an unverified audit obligation.")
        return self
