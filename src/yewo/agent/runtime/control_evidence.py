"""Run-scoped visible-control observations returned by the action verifier."""

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.actuation import VisibleControl
from yewo.artifacts import RecordIdentifier


class VisibleControlObservation(BaseModel):
    """Controls reported for one exact current-run observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: RecordIdentifier
    proposal_id: RecordIdentifier
    controls: tuple[VisibleControl, ...] = Field(min_length=1, max_length=100)


def visible_control_observation(
    observation_id: RecordIdentifier,
    proposal_id: RecordIdentifier,
    controls: tuple[VisibleControl, ...],
) -> VisibleControlObservation | None:
    """Build an observation only when the verifier reported controls."""
    if not controls:
        return None
    return VisibleControlObservation(
        observation_id=observation_id,
        proposal_id=proposal_id,
        controls=controls,
    )


def retain_visible_controls(
    retained: list[VisibleControlObservation],
    observation_id: RecordIdentifier,
    proposal_id: RecordIdentifier,
    controls: tuple[VisibleControl, ...],
) -> None:
    """Append one non-empty verifier observation to the run evidence."""
    observation = visible_control_observation(observation_id, proposal_id, controls)
    if observation is not None:
        retained.append(observation)
