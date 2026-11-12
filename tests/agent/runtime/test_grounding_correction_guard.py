"""Offline tests for re-grounding a target whose recorded point missed."""

from pathlib import Path
from typing import cast

from yewo.agent.actuation import (
    ActionKind,
    ActionProposal,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionRefusalCode,
)
from yewo.agent.actuation.proposal_verifier import (
    ActionReadinessAssessment,
    ActionReadinessDisposition,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.planning.records import PlanningRequest
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.agent.runtime.progress import RunProgressLedger, correct_repeated_grounding
from yewo.artifacts import ArtifactRetention
from yewo.observations import Observation

from yewo.agent import AgentTerminalReason

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)

_DRAWER_ITEM = "the navigation item named 'Edit Record', identified by item text in drawer"


def _tracker() -> ActionNoProgressTracker:
    return ActionNoProgressTracker(ScreenComparisonPolicy())


def _no_change_outcome() -> ActionOutcome:
    return ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        source_observation_id="obs_1",
        settled_observation_id="obs_2",
        action=ActionKind.TAP,
        disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        reason="The settled observation reports no action-outcome signal.",
        signals=frozenset(),
        retention=ArtifactRetention.RUN,
    )


def _record_missed_tap(
    tracker: ActionNoProgressTracker,
    screen: Observation,
    proposal_id: str = "proposal-1",
) -> None:
    tracker.record_outcome(
        proposal(ActionKind.TAP, proposal_id=proposal_id, target_expression=_DRAWER_ITEM),
        screen,
        screen,
        _no_change_outcome(),
    )


def test_a_correction_reusing_the_same_expression_is_no_longer_a_repeat() -> None:
    tracker = _tracker()
    screen = _observation("obs_1", "0000000000000000")
    _record_missed_tap(tracker, screen)

    correction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-2",
        grounding_correction_for_proposal_id="proposal-1",
        target_expression=_DRAWER_ITEM,
    )

    assert tracker.refusal_for(correction, screen) is None


def test_a_plain_repeat_names_the_attempt_a_correction_may_re_ground() -> None:
    tracker = _tracker()
    screen = _observation("obs_1", "0000000000000000")
    _record_missed_tap(tracker, screen)

    repeat = proposal(ActionKind.TAP, proposal_id="proposal-2", target_expression=_DRAWER_ITEM)
    refusal = tracker.refusal_for(repeat, screen)

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS
    assert "grounding_correction_for_proposal_id to proposal-1" in refusal.reason


def test_a_correction_naming_an_attempt_that_never_failed_stays_refused() -> None:
    tracker = _tracker()
    screen = _observation("obs_1", "0000000000000000")
    _record_missed_tap(tracker, screen)

    correction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-2",
        grounding_correction_for_proposal_id="proposal-absent",
        target_expression=_DRAWER_ITEM,
    )
    refusal = tracker.refusal_for(correction, screen)

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS


def test_a_correction_is_accepted_when_the_named_attempt_failed_on_this_screen() -> None:
    """The same attempt also failing elsewhere does not disqualify this screen's correction."""
    tracker = _tracker()
    drawer = _observation("obs_1", "0000000000000000")
    other_screen = _observation("obs_2", "ffffffffffffffff", offset_seconds=1)
    _record_missed_tap(tracker, drawer)
    _record_missed_tap(tracker, other_screen)

    correction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-2",
        grounding_correction_for_proposal_id="proposal-1",
        target_expression=_DRAWER_ITEM,
    )

    assert tracker.refusal_for(correction, drawer) is None


def test_a_correction_for_a_failure_on_another_screen_stays_refused() -> None:
    """What the guard is for: a correction has to answer a failure on the screen it is on.

    The named attempt failed somewhere else. Accepting it here would let one failed tap excuse
    a repeat on every screen the run later reaches.
    """
    tracker = _tracker()
    drawer = _observation("obs_1", "0000000000000000")
    other_screen = _observation("obs_2", "ffffffffffffffff", offset_seconds=1)
    _record_missed_tap(tracker, drawer, "proposal-1")
    _record_missed_tap(tracker, other_screen, "proposal-9")

    correction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-2",
        grounding_correction_for_proposal_id="proposal-9",
        target_expression=_DRAWER_ITEM,
    )
    refusal = tracker.refusal_for(correction, drawer)

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS


def test_the_named_attempt_comes_from_this_screen_not_a_later_failure_elsewhere() -> None:
    tracker = _tracker()
    drawer = _observation("obs_1", "0000000000000000")
    other_screen = _observation("obs_2", "ffffffffffffffff", offset_seconds=1)
    _record_missed_tap(tracker, drawer, "proposal-1")
    _record_missed_tap(tracker, other_screen, "proposal-3")

    repeat = proposal(ActionKind.TAP, proposal_id="proposal-4", target_expression=_DRAWER_ITEM)
    refusal = tracker.refusal_for(repeat, drawer)

    assert refusal is not None
    assert "to proposal-1 " in refusal.reason


def test_the_runtime_re_grounds_a_repeat_then_retires_the_route(tmp_path: Path) -> None:
    stuck = [
        _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
        for index in range(40)
    ]
    planner = SequencePlanner([proposal() for _ in range(5)])

    result = _runtime(tmp_path, planner, ObservationSequence(stuck), RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    corrections = [step.proposal.grounding_correction_for_proposal_id for step in result.steps]
    refusals = [step.refusal.code.value if step.refusal else None for step in result.steps]

    assert corrections[0] is None
    assert corrections[1] is not None
    assert refusals[2] == "repeated_no_progress"
    assert result.terminal_reason is not AgentTerminalReason.CANNOT_PROGRESS


class _CountingProceedVerifier:
    def __init__(self) -> None:
        self.proposals: list[ActionProposal] = []

    def required_model_requests(self, candidate: ActionProposal) -> int:
        return int(candidate.action is ActionKind.TAP)

    def verify(
        self,
        planning_request: PlanningRequest,
        candidate: ActionProposal,
    ) -> ActionReadinessAssessment:
        del planning_request
        self.proposals.append(candidate)
        return ActionReadinessAssessment(
            verification_id=f"verification-{len(self.proposals)}",
            proposal_id=candidate.proposal_id,
            disposition=ActionReadinessDisposition.ALLOW,
            reason="The proposed navigation is ready.",
            completed_work_items=(),
            proposal_work_item=None,
            restarts_completed_work_item=None,
        )


def test_deterministic_repeat_gate_runs_before_the_discovery_verifier(tmp_path: Path) -> None:
    stuck = [
        _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
        for index in range(40)
    ]
    planner = SequencePlanner([proposal() for _ in range(5)])
    verifier = _CountingProceedVerifier()

    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence(stuck),
        RuntimeDevice(),
        proposal_verifier=verifier,
    ).run(
        _task().model_copy(update={"discovery_mode": True}),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert len(verifier.proposals) == 2
    assert verifier.proposals[0].grounding_correction_for_proposal_id is None
    assert verifier.proposals[1].grounding_correction_for_proposal_id is not None
    assert result.steps[2].refusal is not None
    assert result.steps[2].refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS


class _FixedCorrectableLedger:
    """Name one correctable attempt without a persisted progress ledger."""

    def __init__(self, proposal_id: str) -> None:
        self._proposal_id = proposal_id

    def correctable_grounding_proposal_id(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> str | None:
        del proposal, observation
        return self._proposal_id


def test_a_proposal_is_never_declared_a_correction_of_itself() -> None:
    screen = _observation("obs_1", "0000000000000000")
    repeat = proposal(ActionKind.TAP, proposal_id="proposal-1", target_expression=_DRAWER_ITEM)

    corrected, system_declared = correct_repeated_grounding(
        cast(RunProgressLedger, _FixedCorrectableLedger("proposal-1")),
        repeat,
        screen,
        PlannerDecisionSource.MODEL,
    )

    assert system_declared is False
    assert corrected.grounding_correction_for_proposal_id is None
