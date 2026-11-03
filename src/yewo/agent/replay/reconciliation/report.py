"""Terminal report contract for one procedure-compilation attempt."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.models import ModelCallDisposition


class ProcedureCompilationStatus(StrEnum):
    """Terminal status of one non-executing procedure-compilation attempt."""

    COMPILED = "compiled"
    COMPILER_REFUSED = "compiler_refused"
    MODEL_FAILED = "model_failed"
    INVALID_RESPONSE = "invalid_response"


class ProcedureCompilationReport(BaseModel):
    """References and measurements for one procedure-compilation attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
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
