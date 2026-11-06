"""Fail-closed perturbations for compiler-approved procedure replay."""

import json

import pytest
from pydantic import ValidationError
from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.replay.procedure.execution import _procedure_terminal
from yewo.agent.replay.procedure.records import (
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.procedure.store import ProcedureReplayStore
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)
from yewo.agent.runtime.application_state import observation_terminal
from yewo.agent.runtime.records import AgentStep
from yewo.observations import Observation

from agent.discovery.support import agent_result
from agent.replay.support import (
    evidence as _evidence,
    observations as _observations,
    response as _response,
)


def _source() -> tuple[CompiledProcedureActionSource, ReplayProcedure]:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    return CompiledProcedureActionSource(procedure, evidence), procedure


def _request(
    observation: Observation,
    history: tuple[PlanningHistoryEntry, ...] = (),
) -> PlanningRequest:
    return PlanningRequest(
        observation=observation,
        objective="Replay the compiled procedure.",
        completion_conditions=("Every compiled expectation passes.",),
        available_actions=frozenset(ActionKind),
        history=history,
    )


def test_system_repair_boundary_requires_no_retained_action_identity() -> None:
    observation = _observations()[0]
    proposal = ActionProposal(
        proposal_id="repair_boundary",
        source_observation_id=observation.observation_id,
        action=ActionKind.WAIT,
        duration_seconds=0.1,
        expected_outcome="A fresh observation is available.",
        reason="Capture the repair boundary observation.",
    )

    step = AgentStep(
        step_id="repair_step",
        step_index=1,
        source_observation_id=observation.observation_id,
        proposal=proposal,
        decision=PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM),
    )

    assert step.decision.reused_procedure_step_ids == ()

    with pytest.raises(ValidationError, match="retained evidence"):
        AgentStep(
            step_id="repair_step",
            step_index=1,
            source_observation_id=observation.observation_id,
            proposal=proposal,
            decision=PlannerDecisionMetadata(
                source=PlannerDecisionSource.COMPILED_PROCEDURE,
                reused_flow_id="procedure_1",
            ),
        )

    with pytest.raises(ValidationError, match="retained evidence"):
        step.model_copy(
            update={
                "decision": PlannerDecisionMetadata(
                    source=PlannerDecisionSource.COMPILED_PROCEDURE,
                    reused_flow_id="procedure_1",
                )
            }
        ).model_validate(
            step.model_dump(mode="python")
            | {
                "decision": {
                    "source": PlannerDecisionSource.COMPILED_PROCEDURE.value,
                    "reused_flow_id": "procedure_1",
                }
            }
        )


def _outcome_entry(
    proposal,
    before: Observation,
    after: Observation,
) -> PlanningHistoryEntry:
    outcome = (
        agent_result(
            before,
            after,
            action=proposal.action,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        .steps[0]
        .outcome
    )
    assert outcome is not None
    return PlanningHistoryEntry(proposal=proposal, outcome=outcome)


def _advance_first_step(
    source: CompiledProcedureActionSource,
) -> tuple[Observation, ActionProposal]:
    before = _observations()[0]
    after = _observations()[2]
    first = source.plan(_request(before))
    second = source.plan(_request(after, (_outcome_entry(first, before, after),)))
    assert second.action is ActionKind.TYPE
    assert source.completed_step_count == 1
    return after, second


def test_wrong_launch_screen_stops_before_dispatch() -> None:
    source, unused = _source()

    terminal = source.plan(_request(_observations()[-1]))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_step_count == 0
    assert source.divergence is not None
    assert source.divergence.guard == "source"
    assert source.divergence.procedure_index == 0


def test_matching_later_source_screen_enters_at_the_earliest_safe_step() -> None:
    source, unused = _source()
    later_source = _observations()[2]

    action = source.plan(_request(later_source))

    assert action.action is ActionKind.TYPE
    assert source.completed_step_count == 1
    assert source.divergence is None


def test_notification_overlay_is_trusted_but_cannot_authorize_a_procedure_action() -> None:
    source, unused = _source()
    recorded = _observations()[0]
    overlay = recorded.model_copy(
        update={
            "observation_id": "notification_overlay",
            "foreground_application_id": "com.android.systemui",
            "platform_context": "notification_shade",
        }
    )

    terminal = source.plan(_request(overlay))

    assert observation_terminal(overlay) is None
    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_step_count == 0
    assert source.divergence is not None
    assert source.divergence.reason.endswith("source_metadata_changed.")


def test_unexpected_keyboard_state_stops_before_pointer_dispatch() -> None:
    source, unused = _source()
    recorded = _observations()[0]
    keyboard_variant = recorded.model_copy(
        update={
            "observation_id": "unexpected_keyboard",
            "keyboard_visible": not bool(recorded.keyboard_visible),
        }
    )

    terminal = source.plan(_request(keyboard_variant))

    assert observation_terminal(keyboard_variant) is None
    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_step_count == 0
    assert source.divergence is not None
    assert source.divergence.reason.endswith("source_metadata_changed.")


def test_crash_is_application_failure_and_cannot_be_relabelled_as_divergence() -> None:
    source, unused = _source()
    recorded = _observations()[0]
    crashed = recorded.model_copy(
        update={
            "observation_id": "crashed_application",
            "application_state": ApplicationState.NOT_RUNNING,
            "foreground_application_id": None,
        }
    )

    terminal_reason = observation_terminal(crashed)

    assert terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert (
        _procedure_terminal(terminal_reason, source, None, None, None)
        is ProcedureReplayTerminal.APPLICATION_FAILURE
    )
    assert source.divergence is None


def test_relaunch_to_an_earlier_screen_cannot_resume_at_a_later_index() -> None:
    source, unused = _source()
    before, second = _advance_first_step(source)
    relaunched = _observations()[0].model_copy(update={"observation_id": "relaunch_launch_screen"})

    terminal = source.plan(_request(relaunched, (_outcome_entry(second, before, relaunched),)))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_step_count == 1
    assert source.divergence is not None
    assert source.divergence.guard == "expectation"
    assert source.divergence.procedure_index == 1


def test_successful_crash_relaunch_reenters_from_the_matching_screen() -> None:
    source, unused = _source()
    before, second = _advance_first_step(source)
    relaunched = _observations()[0].model_copy(update={"observation_id": "relaunch_entry"})
    feedback = _outcome_entry(second, before, relaunched).model_copy(
        update={
            "application_failure": ActionApplicationFailure(
                classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
                failure_kinds=(ApplicationFailureKind.CRASH,),
                summary="The application crashed and was relaunched.",
                recovery_succeeded=True,
            )
        }
    )

    resumed = source.plan(_request(relaunched, (feedback,)))

    assert resumed.action is ActionKind.TAP
    assert source.decision_metadata().reused_segment_index == 0
    assert source.divergence is None


def test_crash_relaunch_discards_pre_crash_completion_and_milestones() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response(milestones={0: 1}))
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)
    before, second = _advance_first_step(source)
    assert len(source.milestones) == 1
    relaunched = _observations()[0].model_copy(update={"observation_id": "relaunch_entry"})
    feedback = _outcome_entry(second, before, relaunched).model_copy(
        update={
            "application_failure": ActionApplicationFailure(
                classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
                failure_kinds=(ApplicationFailureKind.CRASH,),
                summary="The application crashed and was relaunched.",
                recovery_succeeded=True,
            )
        }
    )

    resumed = source.plan(_request(relaunched, (feedback,)))

    assert resumed.action is ActionKind.TAP
    assert source.completed_step_count == 0
    assert source.milestones == ()


def test_mid_procedure_divergence_is_persisted_and_never_a_completion(
    tmp_path: Path,
) -> None:
    source, procedure = _source()
    before, second = _advance_first_step(source)
    wrong_destination = _observations()[-1].model_copy(
        update={"observation_id": "wrong_mid_procedure_destination"}
    )
    terminal = source.plan(
        _request(
            wrong_destination,
            (_outcome_entry(second, before, wrong_destination),),
        )
    )
    assert terminal.action is ActionKind.CANNOT_PROGRESS
    divergence = source.divergence
    assert divergence is not None
    stopped = agent_result(
        before,
        wrong_destination,
        action=second.action,
        terminal=AgentTerminalReason.CANNOT_PROGRESS,
    )
    result = ProcedureReplayResult(
        procedure_id=procedure.procedure_id,
        evidence_id=procedure.evidence_id,
        source_run_id=procedure.source_run_id,
        setup_id="setup_disturbance",
        application=procedure.application,
        terminal=ProcedureReplayTerminal.DIVERGED,
        procedure_step_count=len(procedure.steps),
        completed_step_count=source.completed_step_count,
        failed_procedure_index=source.failed_procedure_index,
        divergences=(divergence,),
        reason=divergence.reason,
        agent_result=stopped,
    )

    paths = ProcedureReplayStore(tmp_path).persist(result)

    assert tuple(path.name for path in paths) == ("attempt.json", "divergences.json")
    attempt = json.loads((tmp_path / paths[0]).read_text())
    persisted_divergence = json.loads((tmp_path / paths[1]).read_text())
    assert attempt["terminal"] == "diverged"
    assert attempt["verification"] is None
    assert persisted_divergence[0]["procedure_index"] == 1
    assert persisted_divergence[0]["guard"] == "expectation"
