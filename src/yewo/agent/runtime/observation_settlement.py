"""Post-action observation settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.evidence import (
    PlannerDecisionMetadata,
    first_source_of,
)
from yewo.agent.runtime.records import AgentModePolicy
from yewo.observations import Observation, ObservationService, fingerprint_distance
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted


@dataclass(frozen=True)
class SettlementCapture:
    """A settled observation and every observation captured while waiting."""

    observation: Observation
    observation_ids: tuple[str, ...]
    settled: bool


class SettlementInterrupted(RunWorkflowInterrupted):
    """An interruption that preserves observations already captured."""

    def __init__(self, observation_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.observation_ids = observation_ids


class ActionSettlementPolicy(BaseModel):
    """Bounded capture cadence for one dispatched action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_observations: int = Field(ge=2, le=20)
    initial_delay_seconds: float = Field(ge=0, le=5)
    interval_seconds: float = Field(ge=0, le=5)
    required_stable_observations: int = Field(default=2, ge=1, le=2)
    minimum_observations: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_observation_bounds(self) -> ActionSettlementPolicy:
        """Keep the minimum capture count within the hard observation cap."""
        if self.minimum_observations > self.maximum_observations:
            raise ValueError("The minimum observation count cannot exceed the maximum.")
        return self


@runtime_checkable
class ActionSettlementPolicySource(Protocol):
    """Select one action-specific settlement policy from durable evidence."""

    def resolve(
        self,
        default: ActionSettlementPolicy,
        decision: PlannerDecisionMetadata,
    ) -> ActionSettlementPolicy: ...


def default_action_settlement_policy(policy: AgentModePolicy) -> ActionSettlementPolicy:
    """Preserve the configured settlement behavior for ordinary actions."""
    return ActionSettlementPolicy(
        maximum_observations=policy.maximum_settlement_observations,
        initial_delay_seconds=0,
        interval_seconds=policy.settlement_interval_seconds,
    )


def maximum_settlement_window_seconds(policy: ActionSettlementPolicy) -> float:
    """Return the maximum configured sleep time for one settlement attempt."""
    return policy.initial_delay_seconds + policy.interval_seconds * (
        policy.maximum_observations - 1
    )


def resolve_action_settlement_policy(
    mode_policy: AgentModePolicy,
    source: object,
    decision: PlannerDecisionMetadata,
    action: ActionKind,
) -> ActionSettlementPolicy:
    """Resolve one policy without allowing evidence to weaken hard bounds."""
    default = default_action_settlement_policy(mode_policy)
    policy_source = first_source_of(source, ActionSettlementPolicySource)
    resolved = default if policy_source is None else policy_source.resolve(default, decision)
    if resolved.maximum_observations > default.maximum_observations:
        raise ValueError("An action settlement policy cannot raise the observation cap.")
    if action is ActionKind.FOCUS:
        return resolved.model_copy(
            update={"minimum_observations": min(3, resolved.maximum_observations)}
        )
    return resolved


def capture_settled_observation(
    observations: ObservationService,
    source: Observation,
    policy: ActionSettlementPolicy,
    interruption: InterruptionSignal,
    sleep: Callable[[float], None],
) -> SettlementCapture:
    """Wait for a stable post-action screen without accepting the source too early."""
    captured: list[Observation] = []
    observation_ids: list[str] = []
    for attempt_index in range(policy.maximum_observations):
        _require_continuation(interruption, observation_ids)
        delay_seconds = (
            policy.initial_delay_seconds if attempt_index == 0 else policy.interval_seconds
        )
        if delay_seconds > 0:
            sleep(delay_seconds)
            _require_continuation(interruption, observation_ids)
        observation = observations.capture(predecessor_observation_id=source.observation_id)
        captured.append(observation)
        observation_ids.append(observation.observation_id)
        final_attempt = attempt_index + 1 == policy.maximum_observations
        if (
            latest_settlement_candidate(
                source,
                captured,
                policy.required_stable_observations,
                minimum_observations=policy.minimum_observations,
                final_attempt=final_attempt,
            )
            is not None
        ):
            return SettlementCapture(observation, tuple(observation_ids), True)
    if not captured:  # The policy requires at least two attempts.
        raise RuntimeError("Settlement captured no observations.")
    return SettlementCapture(captured[-1], tuple(observation_ids), False)


def latest_settlement_candidate(
    source: Observation,
    captured: list[Observation] | tuple[Observation, ...],
    required_stable_observations: int = 2,
    *,
    minimum_observations: int = 1,
    final_attempt: bool = False,
) -> Observation | None:
    """Accept a stable adjacent state or an exact recurring application image."""
    if len(captured) < minimum_observations:
        return None
    current = captured[-1]
    adjacent = len(captured) >= 2 and observations_are_compatible(captured[-2], current)
    recurring = any(observations_are_identical(previous, current) for previous in captured[:-1])
    stable = required_stable_observations == 1 or adjacent or recurring
    changed_from_source = not observations_are_identical(source, current)
    return current if stable and (changed_from_source or final_attempt) else None


def observations_are_compatible(first: Observation, second: Observation) -> bool:
    """Return whether two observations are visually equivalent for settlement."""
    if not _observation_metadata_matches(first, second):
        return False
    try:
        return fingerprint_distance(first.screen_fingerprint, second.screen_fingerprint) <= 1
    except ValueError:
        return False


def observations_are_identical(first: Observation, second: Observation) -> bool:
    """Return whether an action source and capture contain identical application pixels."""
    return _observation_metadata_matches(first, second) and (
        first.screen_fingerprint.masked_pixels_sha256
        == second.screen_fingerprint.masked_pixels_sha256
    )


def _observation_metadata_matches(first: Observation, second: Observation) -> bool:
    return (
        first.device == second.device
        and first.application == second.application
        and first.application_state is second.application_state
        and first.foreground_application_id == second.foreground_application_id
        and first.platform_context == second.platform_context
        and first.keyboard_visible == second.keyboard_visible
        and first.orientation is second.orientation
        and first.screen_fingerprint.algorithm == second.screen_fingerprint.algorithm
        and first.screen_fingerprint.system_region_mask_sha256
        == second.screen_fingerprint.system_region_mask_sha256
    )


def _require_continuation(interruption: InterruptionSignal, observation_ids: list[str]) -> None:
    if interruption.is_set():
        raise SettlementInterrupted(tuple(observation_ids))
