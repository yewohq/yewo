"""Runtime integration for discovery action readiness checks."""

from pathlib import Path

from yewo.agent.action_verifier import (
    ActionReadinessAssessment,
    ActionReadinessDisposition,
)
from yewo.agent.actuation import ActionKind, ActionProposal, ActionRefusalCode
from yewo.agent.planner import PlanningRequest

from agent.actuation.support import proposal
from agent.test_runtime import (
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
