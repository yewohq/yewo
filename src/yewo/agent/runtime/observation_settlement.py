"""Post-action observation settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.evidence import (
    PlannerDecisionMetadata,
    first_source_of,
)
from yewo.agent.runtime.records import AgentModePolicy
from yewo.observations import (
    Observation,
    ObservationProbe,
    ScreenFingerprint,
    VisualSettlementProbe,
    fingerprint_distance,
)
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted
from yewo.run_narration import current_narrator


@dataclass(frozen=True)
class SettlementCapture:
    """A settled observation and every observation captured while waiting."""

    observation: Observation
    observation_ids: tuple[str, ...]
    is_settled: bool


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


class SettlementObservationService(Protocol):
    """Capture lightweight probes and one persisted settlement observation."""

    def capture_settlement_probe(self) -> VisualSettlementProbe: ...

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation: ...


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
    resolved_policy = default if policy_source is None else policy_source.resolve(default, decision)
    if resolved_policy.maximum_observations > default.maximum_observations:
        raise ValueError("An action settlement policy cannot raise the observation cap.")
    if action is ActionKind.FOCUS:
        return resolved_policy.model_copy(
            update={"minimum_observations": min(3, resolved_policy.maximum_observations)}
        )
    return resolved_policy


def capture_settled_observation(
    observations: SettlementObservationService,
    source: Observation,
    policy: ActionSettlementPolicy,
    interruption: InterruptionSignal,
    sleep: Callable[[float], None],
) -> SettlementCapture:
    """Wait for a stable post-action screen without accepting the source too early."""
    probes: list[VisualSettlementProbe] = []
    for attempt_index in range(policy.maximum_observations):
        _require_continuation(interruption, [])
        delay_seconds = (
            policy.initial_delay_seconds if attempt_index == 0 else policy.interval_seconds
        )
        if delay_seconds > 0:
            _sleep_for_settlement(sleep, delay_seconds, attempt_index)
            _require_continuation(interruption, [])
        probe = _capture_probe(observations, attempt_index)
        probes.append(probe)
        final_attempt = attempt_index + 1 == policy.maximum_observations
        settled = latest_settlement_probe(
            VisualSettlementProbe.from_observation(source),
            probes,
            policy.required_stable_observations,
            minimum_observations=policy.minimum_observations,
            final_attempt=final_attempt,
        )
        if settled or final_attempt:
            observation = observations.capture(predecessor_observation_id=source.observation_id)
            return SettlementCapture(
                observation,
                (observation.observation_id,),
                settled and _probe_matches_observation(probe, observation),
            )
    raise RuntimeError("Settlement captured no probes.")


def latest_settlement_probe(
    source: VisualSettlementProbe,
    probes: list[VisualSettlementProbe] | tuple[VisualSettlementProbe, ...],
    required_stable_observations: int = 2,
    *,
    minimum_observations: int = 1,
    final_attempt: bool = False,
) -> bool:
    """Return whether lightweight probes establish one stable post-action screen."""
    if len(probes) < minimum_observations:
        return False
    current = probes[-1]
    adjacent = len(probes) >= 2 and _settlement_probes_are_compatible(probes[-2], current)
    recurring = any(_settlement_probes_are_identical(previous, current) for previous in probes[:-1])
    stable = required_stable_observations == 1 or adjacent or recurring
    changed_from_source = not _settlement_probes_are_identical(source, current)
    return stable and (changed_from_source or final_attempt)


def latest_settlement_candidate(
    source: Observation,
    captured_observations: list[Observation] | tuple[Observation, ...],
    required_stable_observations: int = 2,
    *,
    minimum_observations: int = 1,
    final_attempt: bool = False,
) -> Observation | None:
    """Accept a stable adjacent state or an exact recurring application image."""
    if len(captured_observations) < minimum_observations:
        return None
    current = captured_observations[-1]
    adjacent = len(captured_observations) >= 2 and observations_are_compatible(
        captured_observations[-2], current
    )
    recurring = any(
        observations_are_identical(previous, current) for previous in captured_observations[:-1]
    )
    stable = required_stable_observations == 1 or adjacent or recurring
    changed_from_source = not observations_are_identical(source, current)
    return current if stable and (changed_from_source or final_attempt) else None


def observations_are_compatible(first: Observation, second: Observation) -> bool:
    """Return whether two observations are visually equivalent for settlement."""
    return _probes_are_compatible(
        ObservationProbe.from_observation(first),
        ObservationProbe.from_observation(second),
    )


def _fingerprints_are_compatible(first: ScreenFingerprint, second: ScreenFingerprint) -> bool:
    if first.algorithm != second.algorithm:
        return False
    if first.system_region_mask_sha256 != second.system_region_mask_sha256:
        return False
    try:
        return fingerprint_distance(first, second) <= 1
    except ValueError:
        return False


def _fingerprints_are_identical(first: ScreenFingerprint, second: ScreenFingerprint) -> bool:
    return (
        first.algorithm == second.algorithm
        and first.system_region_mask_sha256 == second.system_region_mask_sha256
        and first.masked_pixels_sha256 == second.masked_pixels_sha256
    )


def _probe_metadata_matches(first: ObservationProbe, second: ObservationProbe) -> bool:
    return (
        first.device == second.device
        and first.application == second.application
        and first.application_state is second.application_state
        and first.foreground_application_id == second.foreground_application_id
        and first.platform_context == second.platform_context
        and first.keyboard_visible == second.keyboard_visible
        and first.orientation is second.orientation
        and first.screenshot_to_driver_scale == second.screenshot_to_driver_scale
        and first.width_px == second.width_px
        and first.height_px == second.height_px
    )


def _probes_are_compatible(first: ObservationProbe, second: ObservationProbe) -> bool:
    return _probe_metadata_matches(first, second) and _fingerprints_are_compatible(
        first.fingerprint, second.fingerprint
    )


def _probes_are_identical(first: ObservationProbe, second: ObservationProbe) -> bool:
    return _probe_metadata_matches(first, second) and _fingerprints_are_identical(
        first.fingerprint, second.fingerprint
    )


def _probe_matches_observation(
    probe: VisualSettlementProbe,
    observation: Observation,
) -> bool:
    return _settlement_probes_are_compatible(
        probe,
        VisualSettlementProbe.from_observation(observation),
    )


def _settlement_probe_metadata_matches(
    first: VisualSettlementProbe,
    second: VisualSettlementProbe,
) -> bool:
    return (
        first.device == second.device
        and first.application == second.application
        and first.screenshot_to_driver_scale == second.screenshot_to_driver_scale
        and first.width_px == second.width_px
        and first.height_px == second.height_px
    )


def _settlement_probes_are_compatible(
    first: VisualSettlementProbe,
    second: VisualSettlementProbe,
) -> bool:
    return _settlement_probe_metadata_matches(first, second) and _fingerprints_are_compatible(
        first.fingerprint, second.fingerprint
    )


def _settlement_probes_are_identical(
    first: VisualSettlementProbe,
    second: VisualSettlementProbe,
) -> bool:
    return _settlement_probe_metadata_matches(first, second) and _fingerprints_are_identical(
        first.fingerprint, second.fingerprint
    )


def observations_are_identical(first: Observation, second: Observation) -> bool:
    """Return whether an action source and capture contain identical application pixels."""
    return _probes_are_identical(
        ObservationProbe.from_observation(first),
        ObservationProbe.from_observation(second),
    )


def _require_continuation(interruption: InterruptionSignal, observation_ids: list[str]) -> None:
    if interruption.is_set():
        raise SettlementInterrupted(tuple(observation_ids))


def _sleep_for_settlement(
    sleep: Callable[[float], None],
    delay_seconds: float,
    attempt_index: int,
) -> None:
    started = perf_counter()
    sleep(delay_seconds)
    narrator = current_narrator()
    if narrator is not None:
        narrator.trace(
            "device-agent",
            f"settlement_sleep_attempt={attempt_index + 1} "
            f"requested_ms={delay_seconds * 1_000:.1f} "
            f"duration_ms={(perf_counter() - started) * 1_000:.1f}",
        )


def _capture_probe(
    observations: SettlementObservationService,
    attempt_index: int,
) -> VisualSettlementProbe:
    started = perf_counter()
    probe = observations.capture_settlement_probe()
    narrator = current_narrator()
    if narrator is not None:
        narrator.trace(
            "device-agent",
            f"settlement_probe_attempt={attempt_index + 1} "
            f"duration_ms={(perf_counter() - started) * 1_000:.1f}",
        )
    return probe
