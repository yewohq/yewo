"""Structured candidate findings and deterministic confirmation."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.project_config import CoverageCategory
from yewo.reporting import (
    AuditFindingRecord,
    AuditFindingRegionRecord,
    AuditFindingVerificationKind,
    AuditObligationStatus,
)


class ArithmeticRelation(StrEnum):
    """Supported relationships between visible numeric values."""

    SUM = "sum"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    RATIO_PERCENT = "ratio_percent"


class AuditFindingRegionOutput(BaseModel):
    """One normalized region that supports a candidate finding."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation_id: RecordIdentifier
    label: str = Field(min_length=1, max_length=200)
    left: int = Field(ge=0, le=1_000)
    top: int = Field(ge=0, le=1_000)
    right: int = Field(ge=0, le=1_000)
    bottom: int = Field(ge=0, le=1_000)

    @model_validator(mode="after")
    def _validate_positive_area(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("A finding region must have positive area.")
        return self


class ArithmeticFindingVerification(BaseModel):
    """Visible values that code can evaluate as one arithmetic relation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["arithmetic"]
    relation: ArithmeticRelation
    operands: tuple[Decimal, ...] = Field(min_length=2, max_length=20)
    displayed_result: Decimal
    precision_places: int = Field(default=2, ge=0, le=6)
    unit: str | None = Field(default=None, min_length=1, max_length=40)


class EqualityFindingVerification(BaseModel):
    """Two visible or action-backed values that code can compare."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["equality"]
    expected_text: str = Field(min_length=1, max_length=1_000)
    observed_text: str = Field(min_length=1, max_length=1_000)


class VisualFindingVerification(BaseModel):
    """A visible defect that does not need mechanical confirmation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["visual"]


FindingVerification = Annotated[
    ArithmeticFindingVerification | EqualityFindingVerification | VisualFindingVerification,
    Field(discriminator="kind"),
]


class AuditCandidateFindingOutput(BaseModel):
    """One untrusted candidate issue located in supplied evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    outcome_id: RecordIdentifier
    claim: str = Field(min_length=1)
    regions: tuple[AuditFindingRegionOutput, ...] = Field(min_length=1, max_length=20)
    verification: FindingVerification


class ConfirmedAuditFinding(BaseModel):
    """Deterministic disposition of one candidate finding."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    outcome_id: RecordIdentifier
    claim: str = Field(min_length=1)
    is_violation: bool
    reason: str = Field(min_length=1)
    regions: tuple[AuditFindingRegionOutput, ...] = Field(min_length=1, max_length=20)
    verification_kind: Literal["arithmetic", "equality", "visual"]


def build_confirmed_finding_record(finding: ConfirmedAuditFinding) -> AuditFindingRecord:
    """Build the canonical record for one confirmed finding."""
    finding_payload = finding.model_dump_json()
    finding_id = f"finding_{sha256(finding_payload.encode()).hexdigest()[:48]}"
    return AuditFindingRecord(
        finding_id=finding_id,
        claim=finding.claim,
        reason=finding.reason,
        verification_kind=AuditFindingVerificationKind(finding.verification_kind),
        regions=tuple(
            AuditFindingRegionRecord(
                observation_id=region.observation_id,
                label=region.label,
                left=region.left,
                top=region.top,
                right=region.right,
                bottom=region.bottom,
            )
            for region in finding.regions
        ),
    )


def confirm_audit_finding(candidate: AuditCandidateFindingOutput) -> ConfirmedAuditFinding:
    """Confirm a candidate without application-specific rules."""
    verification = candidate.verification
    if isinstance(verification, ArithmeticFindingVerification):
        return _confirm_arithmetic(candidate, verification)
    if isinstance(verification, EqualityFindingVerification):
        expected_text = _normalize_text(verification.expected_text)
        observed_text = _normalize_text(verification.observed_text)
        is_violation = expected_text != observed_text
        relation = "does not match" if is_violation else "matches"
        return ConfirmedAuditFinding(
            outcome_id=candidate.outcome_id,
            claim=candidate.claim,
            is_violation=is_violation,
            reason=(
                f'The observed value "{verification.observed_text}" {relation} '
                f'the expected value "{verification.expected_text}".'
            ),
            regions=candidate.regions,
            verification_kind=verification.kind,
        )
    return ConfirmedAuditFinding(
        outcome_id=candidate.outcome_id,
        claim=candidate.claim,
        is_violation=True,
        reason="The auditor located a directly visible defect in current-run evidence.",
        regions=candidate.regions,
        verification_kind=verification.kind,
    )


def resolve_candidate_findings(
    category: CoverageCategory,
    status: AuditObligationStatus,
    reason: str,
    candidates: tuple[AuditCandidateFindingOutput, ...],
) -> tuple[AuditObligationStatus, str, ConfirmedAuditFinding | None]:
    """Return the final status after deterministic candidate confirmation."""
    applicable = tuple(
        candidate for candidate in candidates if _candidate_applies(category, candidate)
    )
    confirmations = tuple(confirm_audit_finding(candidate) for candidate in applicable)
    violation = next((item for item in confirmations if item.is_violation), None)
    if violation is not None:
        return (
            AuditObligationStatus.UNMET,
            f"{violation.claim} {violation.reason}",
            violation,
        )
    if status is AuditObligationStatus.UNMET and category in {
        CoverageCategory.DATA_CONSISTENCY,
        CoverageCategory.UI_VISUAL,
    }:
        return (
            AuditObligationStatus.UNVERIFIED,
            "The auditor reported a violation without a confirmed candidate.",
            None,
        )
    return status, reason, None


def _candidate_applies(
    category: CoverageCategory,
    candidate: AuditCandidateFindingOutput,
) -> bool:
    verification = candidate.verification
    if category is CoverageCategory.DATA_CONSISTENCY:
        return not isinstance(verification, VisualFindingVerification)
    if category is CoverageCategory.UI_VISUAL:
        return isinstance(verification, VisualFindingVerification)
    return True


def _confirm_arithmetic(
    candidate: AuditCandidateFindingOutput,
    verification: ArithmeticFindingVerification,
) -> ConfirmedAuditFinding:
    expected_value = _evaluate_arithmetic(verification.relation, verification.operands)
    if expected_value is None:
        return ConfirmedAuditFinding(
            outcome_id=candidate.outcome_id,
            claim=candidate.claim,
            is_violation=False,
            reason="The supplied arithmetic relation cannot be evaluated.",
            regions=candidate.regions,
            verification_kind=verification.kind,
        )
    quantum = Decimal(1).scaleb(-verification.precision_places)
    rounded_expected = expected_value.quantize(quantum, rounding=ROUND_HALF_UP)
    rounded_displayed = verification.displayed_result.quantize(
        quantum,
        rounding=ROUND_HALF_UP,
    )
    is_violation = rounded_expected != rounded_displayed
    unit = f" {verification.unit}" if verification.unit else ""
    relation = "does not match" if is_violation else "matches"
    return ConfirmedAuditFinding(
        outcome_id=candidate.outcome_id,
        claim=candidate.claim,
        is_violation=is_violation,
        reason=(
            f"The displayed result {rounded_displayed}{unit} {relation} the calculated "
            f"result {rounded_expected}{unit}."
        ),
        regions=candidate.regions,
        verification_kind=verification.kind,
    )


def _evaluate_arithmetic(
    relation: ArithmeticRelation,
    operands: tuple[Decimal, ...],
) -> Decimal | None:
    if relation is ArithmeticRelation.SUM:
        return sum(operands, start=Decimal(0))
    first, *remaining = operands
    if relation is ArithmeticRelation.DIFFERENCE:
        return first - sum(remaining, start=Decimal(0))
    if relation is ArithmeticRelation.PRODUCT:
        product = Decimal(1)
        for operand in operands:
            product *= operand
        return product
    denominator = operands[1]
    if denominator == 0:
        return None
    return first / denominator * Decimal(100)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())
