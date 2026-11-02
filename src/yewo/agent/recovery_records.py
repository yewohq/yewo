"""Persisted accounting for bounded application recovery."""

from pydantic import BaseModel, ConfigDict, Field

from yewo.artifacts import RecordIdentifier


class RecoveryBudgetCredit(BaseModel):
    """Capped logical allowance restored for Yewo-owned recovery work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: RecordIdentifier | None = None
    steps: int = Field(default=0, ge=0, le=10)
    planning_requests: int = Field(default=0, ge=0, le=11)
    grounding_requests: int = Field(default=0, ge=0, le=20)
    dispatched_actions: int = Field(default=0, ge=0, le=10)
    duration_seconds: float = Field(default=0, ge=0, le=600)
