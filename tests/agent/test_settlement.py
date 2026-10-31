"""Deterministic tests for post-action settlement timing."""

from typing import cast

from yewo.agent.settlement import capture_settled_observation
from yewo.observations import Observation, ObservationService

from agent.test_runtime import NeverInterrupted, _mode_policy, _observation


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
        _mode_policy(settlement_interval_seconds=0.35),
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
        _mode_policy(maximum_settlement_observations=3),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.settled
    assert capture.observation_ids == ("obs_2", "obs_3")
