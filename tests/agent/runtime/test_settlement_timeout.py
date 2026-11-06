"""Post-dispatch observation timeout behavior."""

from pathlib import Path

import pytest

from yewo.observations import Observation, ObservationCaptureError
from yewo.runs import NeverInterrupted

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _stored_steps,
    _task,
)


class FailingSettlementObservations(ObservationSequence):
    """Capture the initial screen, then fail every post-dispatch observation."""

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
        if self.capture_count:
            self.capture_count += 1
            raise ObservationCaptureError("Appium screenshot timed out.")
        return super().capture(predecessor_observation_id=predecessor_observation_id)


def test_settlement_capture_timeout_never_redispatches_the_action(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal()])
    observations = FailingSettlementObservations([_observation("obs_1", "0000000000000000")])
    device = RuntimeDevice()

    with pytest.raises(ObservationCaptureError, match="screenshot timed out"):
        _runtime(tmp_path, planner, observations, device).run(
            _task(),
            _mode_policy(),
            NeverInterrupted(),
        )

    assert len(device.calls) == 1
    assert observations.capture_count == 2
    stored = _stored_steps(tmp_path)
    assert len(stored) == 1
    assert stored[0]["terminal_reason"] == "runtime_failure"
    assert stored[0]["receipt"] is not None
