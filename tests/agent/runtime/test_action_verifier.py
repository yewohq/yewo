"""Runtime integration for discovery action readiness checks."""

import json
from pathlib import Path

from yewo.agent.actuation.verifier import (
    ActionReadinessAssessment,
    ActionReadinessDisposition,
)
from yewo.agent.actuation import ActionKind, ActionProposal, ActionRefusalCode
from yewo.agent.planning.planner import PlanningRequest

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
            unresolved_controls=("The visible Heart field has not been exercised.",),
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
            reason="The floating add button starts another schedule.",
            completed_work_items=(self._create_work,),
            proposal_work_item=self._remaining_work,
            restarts_completed_work_item=self._create_work,
        )


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


def test_completed_work_regression_is_recorded_and_blocked_before_grounding(
    tmp_path: Path,
) -> None:
    create = "Create and save one reusable training schedule."
    reopen = "Return to the saved schedule and establish that it can be repeated or updated."
    planner = SequencePlanner(
        [
            proposal(
                ActionKind.TAP,
                target_expression="the floating add button",
                expected_outcome="A blank schedule form opens.",
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
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["initialized", "assessment"]
