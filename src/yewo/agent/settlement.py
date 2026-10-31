"""Post-action observation settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.runtime_records import AgentModePolicy
from yewo.observations import Observation, ObservationService
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


def capture_settled_observation(
    observations: ObservationService,
    source: Observation,
    policy: AgentModePolicy,
    interruption: InterruptionSignal,
    sleep: Callable[[float], None],
) -> SettlementCapture:
    """Wait for a stable post-action screen without accepting the source too early."""
    previous: Observation | None = None
    observation_ids: list[str] = []
    for attempt_index in range(policy.maximum_settlement_observations):
        _require_continuation(interruption, observation_ids)
        if policy.settlement_interval_seconds > 0:
            sleep(policy.settlement_interval_seconds)
            _require_continuation(interruption, observation_ids)
        observation = observations.capture(predecessor_observation_id=source.observation_id)
        observation_ids.append(observation.observation_id)
        compatible_pair = previous is not None and observations_are_compatible(
            previous, observation
        )
        changed_from_source = not observations_are_compatible(source, observation)
        final_attempt = attempt_index + 1 == policy.maximum_settlement_observations
        if compatible_pair and (changed_from_source or final_attempt):
            return SettlementCapture(observation, tuple(observation_ids), True)
        previous = observation
    if previous is None:  # The policy requires at least two attempts.
        raise RuntimeError("Settlement captured no observations.")
    return SettlementCapture(previous, tuple(observation_ids), False)


def observations_are_compatible(first: Observation, second: Observation) -> bool:
    """Return whether two observations describe the same settled state."""
    return (
        first.device == second.device
        and first.application == second.application
        and first.application_state is second.application_state
        and first.foreground_application_id == second.foreground_application_id
        and first.platform_context == second.platform_context
        and first.keyboard_visible == second.keyboard_visible
        and first.orientation is second.orientation
        and first.screen_fingerprint.algorithm == second.screen_fingerprint.algorithm
        and first.screen_fingerprint.value == second.screen_fingerprint.value
        and first.screen_fingerprint.system_region_mask_sha256
        == second.screen_fingerprint.system_region_mask_sha256
    )


def _require_continuation(interruption: InterruptionSignal, observation_ids: list[str]) -> None:
    if interruption.is_set():
        raise SettlementInterrupted(tuple(observation_ids))
