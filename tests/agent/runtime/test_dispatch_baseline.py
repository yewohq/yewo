"""The runtime compares against the screen as it stood when the action was dispatched."""

from pathlib import Path

from yewo.agent.actuation import ActionKind, ActionOutcomeDisposition
from yewo.observations import Observation, ScreenFingerprint

from agent.actuation.support import proposal
from agent.runtime.support import RuntimeDevice
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


class DriftingScreen(ObservationSequence):
    """A screen that keeps moving on its own while the run is deciding what to do."""

    def __init__(self, observations: list[Observation], settled_fingerprint: str) -> None:
        super().__init__(observations)
        self._settled_fingerprint = settled_fingerprint

    def capture_screen_fingerprint(self) -> ScreenFingerprint:
        current = super().capture_screen_fingerprint()
        return current.model_copy(update={"masked_pixels_sha256": self._settled_fingerprint})


def test_a_screen_that_drifted_before_dispatch_is_not_an_observed_outcome(
    tmp_path: Path,
) -> None:
    # Without a dispatch-time baseline the drift is charged to the action, and the planner is
    # told a tap that did nothing had an observable effect.
    settled = _observation("obs_2", "1111111111111111", offset_seconds=1)
    planner = SequencePlanner([proposal(), proposal(ActionKind.CANNOT_PROGRESS)])
    observations = DriftingScreen(
        [_observation("obs_1", "0000000000000000"), settled, settled],
        settled.screen_fingerprint.masked_pixels_sha256,
    )

    _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    outcome = planner.requests[1].history[0].outcome
    assert outcome is not None
    assert outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
