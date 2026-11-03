"""Deterministic tests for post-action settlement timing."""

from typing import cast

from yewo.agent.runtime.observation_settlement import (
    ActionSettlementPolicy,
    capture_settled_observation,
    default_action_settlement_policy,
)
from yewo.observations import Observation, ObservationService

from agent.runtime.test_agent import NeverInterrupted, _mode_policy, _observation


class OrderedObservations:
    def __init__(self, events: list[str], observations: list[Observation]) -> None:
        self._events = events
        self._observations = iter(observations)

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
        del predecessor_observation_id
        self._events.append("capture")
        return next(self._observations)


def test_settlement_waits_before_its_first_capture() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    observations = OrderedObservations(
        events,
        [
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ],
    )

    capture = capture_settled_observation(
        cast(ObservationService, observations),
        source,
        default_action_settlement_policy(_mode_policy(settlement_interval_seconds=0.35)),
        NeverInterrupted(),
        lambda unused: events.append("sleep"),
    )

    assert events == ["sleep", "capture", "sleep", "capture"]
    assert capture.settled


def test_settlement_detects_exact_change_when_phash_is_unchanged() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    first = _observation("obs_2", "0000000000000000", offset_seconds=1)
    first = first.model_copy(
        update={
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"masked_pixels_sha256": "f" * 64}
            )
        }
    )
    second = first.model_copy(
        update={
            "observation_id": "obs_3",
            "redacted_screenshot": first.redacted_screenshot.model_copy(
                update={"artifact_id": "shot-obs_3"}
            ),
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"source_artifact_id": "shot-obs_3"}
            ),
        }
    )

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [first, second])),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=3)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.settled
    assert capture.observation_ids == ("obs_2", "obs_3")


def test_settlement_accepts_minor_perceptual_variation_between_captures() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    first = _observation("obs_2", "1000000000000000", offset_seconds=1)
    second = _observation("obs_3", "1000000000000001", offset_seconds=2)

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [first, second])),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=3)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.settled
    assert capture.observation_ids == ("obs_2", "obs_3")


def test_action_policy_can_capture_immediately_then_use_its_interval() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    observations = OrderedObservations(
        events,
        [
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ],
    )

    capture = capture_settled_observation(
        cast(ObservationService, observations),
        source,
        ActionSettlementPolicy(
            maximum_observations=6,
            initial_delay_seconds=0,
            interval_seconds=0.2,
        ),
        NeverInterrupted(),
        lambda duration: events.append(f"sleep:{duration}"),
    )

    assert events == ["capture", "sleep:0.2", "capture"]
    assert capture.settled


def test_replay_policy_accepts_its_first_changed_observation() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    changed = _observation("obs_2", "1111111111111111", offset_seconds=1)

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [changed])),
        source,
        ActionSettlementPolicy(
            maximum_observations=6,
            initial_delay_seconds=0,
            interval_seconds=0.2,
            required_stable_observations=1,
        ),
        NeverInterrupted(),
        lambda duration: events.append(f"sleep:{duration}"),
    )

    assert events == ["capture"]
    assert capture.observation_ids == ("obs_2",)
    assert capture.settled
