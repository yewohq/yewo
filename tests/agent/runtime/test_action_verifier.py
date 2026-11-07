"""Runtime integration for discovery action readiness checks."""

import json
from pathlib import Path

import pytest

from yewo.agent.actuation.verifier import (
    ActionReadinessAssessment,
    ActionReadinessDisposition,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionProposal,
    ActionRefusalCode,
    RequiredControlBasis,
    UnresolvedControl,
)
from yewo.agent.planning.records import (
    PlanningFocusCause,
    PlanningFocusContext,
    PlanningRequest,
)
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.verification import verify_discovery_action
from yewo.agent.runtime.no_progress import MAXIMUM_CONSECUTIVE_REFUSALS
from yewo.agent.runtime.failure import ModelServiceUnavailableError

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


class _RefusingActionVerifier:
    def required_model_requests(self, candidate: ActionProposal) -> int:
        return int(candidate.action is ActionKind.TAP)

    def verify(
        self,
        planning_request: PlanningRequest,
        candidate: ActionProposal,
    ) -> ActionReadinessAssessment:
        del planning_request
        return ActionReadinessAssessment(
            verification_id="verification_1",
            proposal_id=candidate.proposal_id,
            disposition=ActionReadinessDisposition.REFUSE,
            reason="Saving would leave visible discovery work unresolved.",
            unresolved_controls=(
                UnresolvedControl(
                    control_name="Heart",
                    required_by=RequiredControlBasis.OBJECTIVE_OR_OPEN_WORK,
                    requirement_evidence="Inspect every relevant visible control.",
                    reason="The visible field has not been exercised.",
                ),
            ),
            completed_work_items=(),
            proposal_work_item=None,
            restarts_completed_work_item=None,
        )


class _UnavailableActionVerifier:
    def required_model_requests(self, candidate: ActionProposal) -> int:
        return int(candidate.action is ActionKind.TAP)

    def verify(
        self,
        planning_request: PlanningRequest,
        candidate: ActionProposal,
    ) -> ActionReadinessAssessment:
        del planning_request
        return ActionReadinessAssessment(
            verification_id=f"verification_{candidate.proposal_id}",
            proposal_id=candidate.proposal_id,
            disposition=ActionReadinessDisposition.UNAVAILABLE,
            reason="The action verifier did not return a usable decision.",
            completed_work_items=(),
            proposal_work_item=None,
            restarts_completed_work_item=None,
        )


class _CompletedWorkRegressionVerifier:
    def __init__(self, create_work: str, remaining_work: str) -> None:
        self._create_work = create_work
        self._remaining_work = remaining_work

    def required_model_requests(self, candidate: ActionProposal) -> int:
        return int(candidate.action is ActionKind.TAP)

    def verify(
        self,
        planning_request: PlanningRequest,
        candidate: ActionProposal,
    ) -> ActionReadinessAssessment:
        progress_context = planning_request.progress_context
        assert progress_context is not None
        assert self._remaining_work in progress_context.remaining_work
        return ActionReadinessAssessment(
            verification_id="verification_regression",
            proposal_id=candidate.proposal_id,
            disposition=ActionReadinessDisposition.REFUSE,
            reason="The visible create control starts another record.",
            completed_work_items=(self._create_work,),
            proposal_work_item=self._remaining_work,
            restarts_completed_work_item=self._create_work,
        )


class _RecordingActionVerifier:
    def __init__(self) -> None:
        self.requests: list[tuple[PlanningRequest, ActionProposal]] = []

    def required_model_requests(self, proposal: ActionProposal) -> int:
        del proposal
        return 0

    def verify(
        self,
        planning_request: PlanningRequest,
        proposal: ActionProposal,
    ) -> ActionReadinessAssessment:
        self.requests.append((planning_request, proposal))
        return ActionReadinessAssessment(
            verification_id="verification-focus",
            proposal_id=proposal.proposal_id,
            disposition=ActionReadinessDisposition.ALLOW,
            reason="The focused proposal follows the no-progress constraints.",
            completed_work_items=(),
            proposal_work_item=None,
            restarts_completed_work_item=None,
        )


def test_focus_respects_verifiers_zero_request_declaration() -> None:
    screen = _observation("obs-focus", "0000000000000000")
    focus = PlanningFocusContext(
        focus_id="focus-1",
        cause=PlanningFocusCause.SCREEN_NO_PROGRESS,
        reason="The screen did not advance.",
        prohibited_interaction_intents=("tap|ref:old-control||||",),
    )
    request = PlanningRequest(
        observation=screen,
        objective="Reach another supported workflow stage.",
        completion_conditions=("A new stage is established.",),
        available_actions=frozenset({ActionKind.FOCUS}),
        planning_focus=focus,
        discovery_mode=True,
    )
    candidate = proposal(
        ActionKind.FOCUS,
        proposal_id="proposal-focus",
        source_observation_id=screen.observation_id,
        target_expression="a different visible field",
    )
    verifier = _RecordingActionVerifier()

    result = verify_discovery_action(
        verifier,
        True,
        PlannerDecisionSource.MODEL,
        object(),
        request,
        candidate,
        _mode_policy(),
        RecoveryRunState(None, ScreenComparisonPolicy()),
        0,
        0.0,
        0,
        0,
        1.0,
    )

    assert result.request_count == 0
    assert verifier.requests == []


def test_discovery_verifier_blocks_premature_save_before_grounding(tmp_path: Path) -> None:
    planner = SequencePlanner(
        [
            proposal(
                ActionKind.TAP,
                target_expression="the visible Save button",
                expected_outcome="The editor saves and closes.",
            ),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])
    device = RuntimeDevice()

    result = _runtime(
        tmp_path,
        planner,
        observations,
        device,
        action_verifier=_RefusingActionVerifier(),
    ).run(
        _task().model_copy(update={"discovery_mode": True}),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.steps[0].refusal is not None
    assert result.steps[0].refusal.code is ActionRefusalCode.DISCOVERY_INCOMPLETE
    assert planner.requests[1].history[0].refusal == result.steps[0].refusal
    assert result.usage.planning_requests == 3
    assert result.usage.grounding_requests == 0
    assert result.usage.dispatched_actions == 0
    assert device.calls == []


def test_unavailable_verifier_does_not_poison_no_progress_state(tmp_path: Path) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TAP, target_expression="the visible Save button"),
            proposal(ActionKind.TAP, target_expression="the visible Save button"),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])
    device = RuntimeDevice()

    result = _runtime(
        tmp_path,
        planner,
        observations,
        device,
        action_verifier=_UnavailableActionVerifier(),
    ).run(
        _task().model_copy(update={"discovery_mode": True}),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert len(result.steps) == 3
    assert [step.refusal.code for step in result.steps[:2] if step.refusal] == [
        ActionRefusalCode.VERIFICATION_UNAVAILABLE,
        ActionRefusalCode.VERIFICATION_UNAVAILABLE,
    ]
    assert result.steps[2].terminal_reason is not None
    assert result.usage.dispatched_actions == 0
    assert device.calls == []


def test_repeated_unavailable_verifier_stops_at_the_refusal_deadlock_bound(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TAP, target_expression="the visible Save button")
            for _ in range(MAXIMUM_CONSECUTIVE_REFUSALS)
        ]
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    device = RuntimeDevice()
    runtime = _runtime(
        tmp_path,
        planner,
        observations,
        device,
        action_verifier=_UnavailableActionVerifier(),
    )

    with pytest.raises(ModelServiceUnavailableError, match="model service remained unavailable"):
        runtime.run(
            _task().model_copy(update={"discovery_mode": True}),
            _mode_policy(),
            NeverInterrupted(),
        )

    assert len(planner.requests) == MAXIMUM_CONSECUTIVE_REFUSALS
    assert device.calls == []


def test_completed_work_regression_is_recorded_and_blocked_before_grounding(
    tmp_path: Path,
) -> None:
    create = "Create and save one reusable record."
    reopen = "Return to the saved record and establish that it can be reopened or updated."
    planner = SequencePlanner(
        [
            proposal(
                ActionKind.TAP,
                target_expression="the visible create control",
                expected_outcome="A blank record form opens.",
            ),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])
    device = RuntimeDevice()

    result = _runtime(
        tmp_path,
        planner,
        observations,
        device,
        action_verifier=_CompletedWorkRegressionVerifier(create, reopen),
    ).run(
        _task().model_copy(
            update={
                "discovery_mode": True,
                "open_work": (create, reopen),
            }
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.steps[0].refusal is not None
    assert result.steps[0].refusal.code is ActionRefusalCode.COMPLETED_WORK_REGRESSION
    assert planner.requests[1].progress_context is not None
    assert planner.requests[1].progress_context.completed_work_items == (create,)
    assert planner.requests[1].open_work == (reopen,)
    assert result.usage.grounding_requests == 0
    assert result.usage.dispatched_actions == 0
    assert device.calls == []
    events = [
        json.loads(line)
        for line in (tmp_path / "agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["initialized", "assessment", "refusal"]
    assert events[-1]["refusal_code"] == "completed_work_regression"
