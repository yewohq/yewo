"""Persist aggregate evidence for Device Agent infrastructure failures."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.runtime.records import AgentStep, AgentUsage
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    ArtifactWriteError,
    AtomicArtifactWriter,
    RecordIdentifier,
)


class AgentFailureSummary(BaseModel):
    """Durable run totals retained when no AgentResult can be returned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    failure_kind: Literal["model_service_unavailable"] = "model_service_unavailable"
    reason: str = Field(min_length=1)
    final_observation_id: RecordIdentifier
    usage: AgentUsage
    step_ids: tuple[RecordIdentifier, ...]


class AgentFailureSummaryPersistenceError(RuntimeError):
    """The runtime could not retain its infrastructure-failure totals."""


class ModelServiceUnavailableError(RuntimeError):
    """A required model service stayed unavailable through the retry bound."""

    def __init__(self, summary: AgentFailureSummary) -> None:
        super().__init__(summary.reason)
        self.summary = summary


def persist_model_service_failure(
    root: Path,
    reason: str,
    final_observation_id: RecordIdentifier,
    steps: list[AgentStep],
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
) -> AgentFailureSummary:
    """Persist one aggregate without converting the failure into an AgentResult."""
    summary = AgentFailureSummary(
        reason=reason,
        final_observation_id=final_observation_id,
        usage=AgentUsage(
            steps=len(steps),
            planning_requests=planning_requests,
            grounding_requests=grounding_requests,
            dispatched_actions=dispatched_actions,
        ),
        step_ids=tuple(step.step_id for step in steps),
    )
    path = ARTIFACTS_DIRECTORY / "agent" / "failure-summary.json"
    try:
        AtomicArtifactWriter(root).write(
            path,
            (summary.model_dump_json() + "\n").encode("utf-8"),
        )
    except ArtifactWriteError as error:
        raise AgentFailureSummaryPersistenceError(
            "Could not persist the Device Agent failure summary."
        ) from error
    return summary
