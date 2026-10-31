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
