"""The runtime asks whether a pixel-only change was the change the action predicted."""

from pathlib import Path

from yewo.agent.actuation import ActionOutcomeDisposition, ActionOutcomeSignal
from yewo.agent.actuation.outcome_verifier import OutcomeAssessment, OutcomeVerdict

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


class ContradictingVerifier:
    """Report that the screen moved for a reason other than the dispatched action."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify(self, action, before_screenshot, after) -> OutcomeAssessment:
        del before_screenshot, after
        self.calls.append(action.authorization_id)
        return OutcomeAssessment(
            assessment_id="assessment-1",
            authorization_id=action.authorization_id,
            verdict=OutcomeVerdict.NOT_HAPPENED,
            reason="The Home tab is still selected and no Favourites list is visible.",
        )


def test_a_pixel_only_change_the_verifier_contradicts_stops_being_observed(
    tmp_path: Path,
) -> None:
    # Without this the planner is told a tap that hit the wrong control had an observable effect.
    verifier = ContradictingVerifier()
    planner = SequencePlanner([proposal(), proposal()])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        predicted_outcome_verifier=verifier,
    ).run(_task(), _mode_policy(maximum_steps=1), NeverInterrupted())

    assert verifier.calls, "the runtime never asked whether the prediction happened"
    outcome = result.steps[0].outcome
    assert outcome is not None
    assert ActionOutcomeSignal.PIXELS_CHANGED in outcome.signals
    assert outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS
    assert "Home tab is still selected" in outcome.reason
