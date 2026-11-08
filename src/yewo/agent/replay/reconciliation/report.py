"""Terminal report contract for one procedure-compilation attempt."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.models import ModelCallDisposition, ModelCallRecord, ModelResponse


class ProcedureCompilationStatus(StrEnum):
    """Terminal status of one non-executing procedure-compilation attempt."""

    COMPILED = "compiled"
    COMPILER_REFUSED = "compiler_refused"
    MODEL_FAILED = "model_failed"
    INVALID_RESPONSE = "invalid_response"


class ProcedureCompilationReport(BaseModel):
    """References and measurements for one procedure-compilation attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_run_id: RecordIdentifier
    evidence_id: RecordIdentifier
    request_id: RecordIdentifier
    status: ProcedureCompilationStatus
    model_disposition: ModelCallDisposition
    model_call_artifact_ids: tuple[RecordIdentifier, ...] = ()
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    procedure_id: RecordIdentifier | None = None
    compiler_refusal_reason_count: int = Field(default=0, ge=0)
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _validate_outcome_reference(self) -> Self:
        if (self.status is ProcedureCompilationStatus.COMPILED) != (self.procedure_id is not None):
            raise ValueError("Only a compiled result can identify a procedure.")
        if (self.status is ProcedureCompilationStatus.COMPILER_REFUSED) != (
            self.compiler_refusal_reason_count > 0
        ):
            raise ValueError("Only a compiler refusal can contain compiler reasons.")
        return self


@dataclass(frozen=True)
class ProcedureCompilationMetrics:
    """Complete usage measurements for one or more reconciliation responses."""

    model_call_artifact_ids: tuple[RecordIdentifier, ...]
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: Decimal | None


def model_response_metrics(response: ModelResponse) -> ProcedureCompilationMetrics:
    """Collect complete metrics from one model response."""
    return _model_call_metrics(response.calls)


def combined_model_response_metrics(
    responses: tuple[ModelResponse, ...],
) -> ProcedureCompilationMetrics:
    """Collect complete metrics across an initial and fallback response."""
    calls = tuple(call for response in responses for call in response.calls)
    return _model_call_metrics(calls)


def build_procedure_compilation_report(
    *,
    run_id: RecordIdentifier,
    evidence_id: RecordIdentifier,
    request_id: RecordIdentifier,
    status: ProcedureCompilationStatus,
    model_disposition: ModelCallDisposition,
    reason: str,
    metrics: ProcedureCompilationMetrics,
    procedure_id: RecordIdentifier | None = None,
    compiler_refusal_reason_count: int = 0,
) -> ProcedureCompilationReport:
    """Build one report from validated outcome references and model metrics."""
    return ProcedureCompilationReport(
        source_run_id=run_id,
        evidence_id=evidence_id,
        request_id=request_id,
        status=status,
        model_disposition=model_disposition,
        reason=reason,
        model_call_artifact_ids=metrics.model_call_artifact_ids,
        latency_ms=metrics.latency_ms,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        reasoning_tokens=metrics.reasoning_tokens,
        cost_usd=metrics.cost_usd,
        procedure_id=procedure_id,
        compiler_refusal_reason_count=compiler_refusal_reason_count,
    )


def _model_call_metrics(
    calls: tuple[ModelCallRecord, ...],
) -> ProcedureCompilationMetrics:
    return ProcedureCompilationMetrics(
        model_call_artifact_ids=tuple(
            call.call_artifact.artifact_id for call in calls if call.call_artifact is not None
        ),
        latency_ms=_complete_latency_sum(calls),
        prompt_tokens=_complete_token_sum(calls, "prompt"),
        completion_tokens=_complete_token_sum(calls, "completion"),
        reasoning_tokens=_complete_token_sum(calls, "reasoning"),
        cost_usd=_complete_cost_sum(calls),
    )


def _complete_latency_sum(calls: tuple[ModelCallRecord, ...]) -> float | None:
    values = tuple(call.latency_ms for call in calls)
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _complete_token_sum(
    calls: tuple[ModelCallRecord, ...],
    metric: Literal["prompt", "completion", "reasoning"],
) -> int | None:
    values_by_metric = {
        "prompt": tuple(call.usage.prompt_tokens for call in calls),
        "completion": tuple(call.usage.completion_tokens for call in calls),
        "reasoning": tuple(call.usage.reasoning_tokens for call in calls),
    }
    values = values_by_metric[metric]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _complete_cost_sum(calls: tuple[ModelCallRecord, ...]) -> Decimal | None:
    values = tuple(call.usage.cost_usd for call in calls)
    if any(value is None for value in values):
        return None
    return sum(
        (value for value in values if value is not None),
        start=Decimal(0),
    )
