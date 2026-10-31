"""Requests and evidence conclusions for YewoAgent discovery."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.runtime_records import AgentResult, AgentTerminalReason
from yewo.agent.route_replay import RouteReplayResult
from yewo.agent.goal_verifier import GoalVerification
from yewo.artifacts import RecordIdentifier


DiscoveryText = Annotated[str, Field(min_length=1, max_length=4_000)]


class DiscoveryObjectiveState(StrEnum):
    """Evidence state of one requested discovery objective."""

    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"
    UNATTEMPTED = "unattempted"


class DiscoveryObjective(BaseModel):
    """One journey-relevant application fact to investigate."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    objective_id: RecordIdentifier
    statement: DiscoveryText
    completion_conditions: tuple[DiscoveryText, ...] = Field(default=(), max_length=100)
    required: bool = True


class DiscoveryRequest(BaseModel):
    """Bounded journey-directed work supplied to discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: RecordIdentifier
    objectives: tuple[DiscoveryObjective, ...] = Field(min_length=1, max_length=1)
    known_facts: tuple[DiscoveryText, ...] = Field(default=(), max_length=100)
    priorities: tuple[DiscoveryText, ...] = Field(default=(), max_length=99)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def _validate_objective_ids(self) -> Self:
        objective_ids = [objective.objective_id for objective in self.objectives]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("Discovery objective identifiers must be unique.")
        if len(self.declared_value_names) != len(set(self.declared_value_names)):
            raise ValueError("Discovery declared-value names must be unique.")
        return self


class DiscoveryCoverageEntry(BaseModel):
    """Truthful conclusion for one requested discovery objective."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    objective_id: RecordIdentifier
    state: DiscoveryObjectiveState
    capability_id: RecordIdentifier | None = None
    route_id: RecordIdentifier | None = None
    evidence_observation_ids: tuple[RecordIdentifier, ...] = ()
    reason: DiscoveryText

    @model_validator(mode="after")
    def _validate_evidence_state(self) -> Self:
        evidence_fields = (
            self.capability_id,
            self.route_id,
            *self.evidence_observation_ids,
        )
        if self.state is DiscoveryObjectiveState.SUPPORTED:
            if self.capability_id is None or self.route_id is None:
                raise ValueError(
                    "Supported discovery coverage requires capability and route evidence."
                )
            if not self.evidence_observation_ids:
                raise ValueError("Supported discovery coverage requires observations.")
        elif any(value is not None for value in evidence_fields[:2]):
            raise ValueError("Unresolved discovery coverage cannot claim reusable evidence.")
        return self


class DiscoveryCoverage(BaseModel):
    """Objective coverage without a complete-application claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[DiscoveryCoverageEntry, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_unique_objectives(self) -> Self:
        objective_ids = [entry.objective_id for entry in self.entries]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("Discovery coverage must contain one entry per objective.")
        return self


class DiscoveryResult(BaseModel):
    """Discovery conclusion and the Device Agent evidence that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: DiscoveryRequest
    terminal_reason: AgentTerminalReason
    coverage: DiscoveryCoverage
    agent_result: AgentResult
    replay: RouteReplayResult | None = None
    verification: GoalVerification | None = None
    memory_id: RecordIdentifier | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        requested = {objective.objective_id for objective in self.request.objectives}
        covered = {entry.objective_id for entry in self.coverage.entries}
        if requested != covered:
            raise ValueError("Discovery coverage must account for every requested objective.")
        if self.terminal_reason is not self.agent_result.terminal_reason:
            raise ValueError("Discovery and Device Agent terminals must match.")
        has_supported = any(
            entry.state is DiscoveryObjectiveState.SUPPORTED for entry in self.coverage.entries
        )
        if has_supported and self.memory_id is None:
            raise ValueError("Supported discovery coverage must identify promoted memory.")
        if self.verification is not None and (
            self.verification.objective_id not in requested
            or self.verification.final_observation_id != self.agent_result.final_observation_id
        ):
            raise ValueError("Discovery verification does not match the result evidence.")
        return self
