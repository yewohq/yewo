"""Shared deterministic evidence for compiled-procedure tests."""

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.actuation import ActionKind
from yewo.agent.replay import (
    ReplayProcedure,
    compile_replay_procedure,
    CompilerCheck,
    CompilerRefusal,
    EvidenceDocument,
    ReconcilerResponse,
    build_evidence_document,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
    ScreenWorkflowRole,
)
from yewo.observations import Observation

from agent.discovery.support import agent_result, observation


def observations() -> tuple[Observation, ...]:
    """Return one deterministic observation chain for procedure compilation."""
    return (
        observation("obs_1", "0000000000000000", offset_seconds=0),
        observation(
            "obs_same",
            "0000000000000001",
            predecessor_observation_id="obs_1",
            offset_seconds=1,
        ),
        observation(
            "obs_2",
            "ffffffffffffffff",
            predecessor_observation_id="obs_same",
            offset_seconds=2,
        ),
        observation(
            "obs_3",
            "aaaaaaaaaaaaaaaa",
            predecessor_observation_id="obs_2",
            offset_seconds=3,
        ),
        observation(
            "obs_4",
            "5555555555555555",
            predecessor_observation_id="obs_3",
            offset_seconds=4,
        ),
    )


def evidence() -> EvidenceDocument:
    """Build reusable evidence with navigation, input, and commit steps."""
    observed = observations()
    first = (
        agent_result(
            observed[0],
            observed[2],
            action=ActionKind.TAP,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        .steps[0]
        .model_copy(update={"step_id": "step_1", "step_index": 1})
    )
    second = (
        agent_result(
            observed[2],
            observed[3],
            action=ActionKind.TYPE,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        .steps[0]
        .model_copy(update={"step_id": "step_2", "step_index": 2})
    )
    third_source = agent_result(
        observed[3],
        observed[4],
        action=ActionKind.TAP,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    third = third_source.model_copy(
        update={
            "step_id": "step_3",
            "step_index": 3,
            "proposal": third_source.proposal.model_copy(
                update={
                    "target_element_ref": "save-control",
                    "target_expression": "the 'Save' button",
                }
            ),
        }
    )
    save_analysis = CurrentScreenAnalysis(
        analysis_id="analysis-save",
        observation_id=observed[3].observation_id,
        screenshot_sha256=observed[3].screenshot.sha256,
        presentation=observed[3].presentation_key,
        screen_summary="Visible form with save control.",
        elements=(
            ScreenElement(
                element_ref="save-control",
                name="Save",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.CONTROL_SURFACE,
                    semantic=ScreenSemanticRole.BUTTON,
                ),
                confidence=ScreenElementConfidence.HIGH,
                workflow_role=ScreenWorkflowRole.ADVANCE_STAGE,
            ),
        ),
    )
    task = AgentTask(
        objective="Create and save one named item.",
        completion_conditions=("The named item is saved.",),
        discovery_mode=True,
    )
    return build_evidence_document(
        run_id="run_1",
        objective_id="objective_1",
        task=task,
        steps=(first, second, third),
        observations=observed,
        screen_analyses=(save_analysis,),
        identifier_factory=lambda: "evidence_1",
    )


def response(
    procedure: tuple[int, ...] = (1, 2, 3),
    milestones: dict[int, int] | None = None,
    completion_conditions: tuple[str, ...] = ("The named item is saved.",),
) -> ReconcilerResponse:
    """Return one valid reconciler response with optional selections."""
    return ReconcilerResponse(
        request_id="request_1",
        completion_conditions=completion_conditions,
        procedure=procedure,
        milestones={0: 3} if milestones is None else milestones,
    )


def checks(refusal: CompilerRefusal) -> set[CompilerCheck]:
    """Return the checks reported by one compiler refusal."""
    return {reason.check for reason in refusal.reasons}


def procedure() -> ReplayProcedure:
    """Compile the shared evidence into one replay procedure."""
    compiled = compile_replay_procedure(evidence(), response())
    assert isinstance(compiled, ReplayProcedure)
    return compiled
