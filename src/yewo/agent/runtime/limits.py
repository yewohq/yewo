"""Absolute and default Device Agent runtime limits."""

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.actuation import ActionPolicyLimits


DEFAULT_MAXIMUM_STEPS = 100
MAXIMUM_STEPS_LIMIT = 500
MAXIMUM_CREDITED_RECOVERY_STEPS = 10
# Recovery refunds its own steps, so the loop can dispatch past the logical limit.
MAXIMUM_PHYSICAL_STEPS = MAXIMUM_STEPS_LIMIT + MAXIMUM_CREDITED_RECOVERY_STEPS
DEFAULT_MAXIMUM_DURATION_SECONDS = 1_200
MAXIMUM_DURATION_SECONDS_LIMIT = 14_400


class AgentModePolicy(BaseModel):
    """Workflow limits applied to one Device Agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_limits: ActionPolicyLimits
    maximum_steps: int = Field(default=DEFAULT_MAXIMUM_STEPS, gt=0, le=MAXIMUM_STEPS_LIMIT)
    maximum_grounding_requests: int = Field(ge=0, le=1_000)
    maximum_model_requests: int | None = Field(default=None, gt=0, le=2_001)
    maximum_duration_seconds: float = Field(
        default=DEFAULT_MAXIMUM_DURATION_SECONDS,
        gt=0,
        le=MAXIMUM_DURATION_SECONDS_LIMIT,
    )
    maximum_settlement_observations: int = Field(default=7, ge=2, le=20)
    settlement_interval_seconds: float = Field(default=0.35, ge=0, le=5)
