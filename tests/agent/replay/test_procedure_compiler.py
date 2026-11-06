"""Deterministic procedure-evidence and reconciliation compiler tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.actuation import ActionKind
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.procedure.grounding import ProcedureGrounder
from yewo.agent.replay.procedure.execution import _procedure_terminal
from yewo.agent.replay.procedure.records import ProcedureReplayTerminal
from yewo.agent.replay.procedure.store import (
    CompiledProcedureStore,
)
from yewo.agent.replay import (
    COMPILER_REFUSAL_PATH,
    REPLAY_PROCEDURE_PATH,
    CompilerCheck,
    CompilerRefusal,
    EvidenceResult,
    EvidenceStepKind,
    ReconcilerDecision,
    ReconcilerResponse,
    ReconciliationArtifactStore,
    ReplayProcedure,
    build_evidence_document,
    compile_replay_procedure,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.grounding import GroundingResolutionSource, ProcedureGroundingResolution

from agent.discovery.support import agent_result
from agent.replay.support import checks as _checks
from agent.replay.support import evidence as _evidence
from agent.replay.support import observations as _observations
from agent.replay.support import response as _response


def test_evidence_is_lossless_groups_states_and_marks_inference() -> None:
    evidence = _evidence()

    assert evidence.run.step_count == 3
    assert evidence.run.observation_count == 5
    assert evidence.states["state_001"].observation_ids == ("obs_1", "obs_same")
    assert evidence.steps[1].control.kind.semantic is ScreenSemanticRole.TEXT_FIELD
    assert evidence.steps[1].control.kind_source == "recorded_action"
    assert evidence.steps[1].control.inference.input_sources == (
        "recorded_action",
        "recorded_input_contract",
    )
    assert len(evidence.diagnostics.data_entry) == 1
    encoded = evidence.model_dump_json()
    assert "accessibility_id" not in encoded
    assert "resource_id" not in encoded
    assert "platform_class" not in encoded


def test_evidence_uses_analyzed_element_kind_instead_of_reparsing_prose() -> None:
    before, _, after, *_ = _observations()
    source = agent_result(
        before,
        after,
        action=ActionKind.TAP,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    step = source.model_copy(
        update={
            "proposal": source.proposal.model_copy(
                update={
                    "target_element_ref": "element-1",
                    "target_expression": "the visible button",
                }
            )
        }
    )
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=before.observation_id,
        screenshot_sha256=before.redacted_screenshot.sha256,
        presentation=before.presentation_key,
        screen_summary="Visible records screen.",
        elements=(
            ScreenElement(
                element_ref="element-1",
                name="Orders",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.TEXT_AND_ICON,
                    semantic=ScreenSemanticRole.NAVIGATION_ITEM,
                ),
                confidence=ScreenElementConfidence.HIGH,
                evidence="A labelled navigation item is visible.",
            ),
        ),
    )

    evidence = build_evidence_document(
        run_id="run-1",
        objective_id="objective-1",
        task=AgentTask(objective="Open orders.", discovery_mode=True),
        steps=(step,),
        observations=(before, after),
        screen_analyses=(analysis,),
        identifier_factory=lambda: "evidence-1",
    )

    control = evidence.steps[0].control
    assert control.kind.semantic is ScreenSemanticRole.NAVIGATION_ITEM
    assert control.kind_source == "screen_analysis"
    assert control.label == "Orders"
    assert control.inference.method == "screen-analysis-element-v1"


def test_compiler_builds_zero_based_procedure_and_capture_point() -> None:
    outcome = compile_replay_procedure(
        _evidence(),
        _response(),
        identifier_factory=lambda: "procedure_1",
    )

    assert isinstance(outcome, ReplayProcedure)
    assert tuple(step.procedure_index for step in outcome.steps) == (0, 1, 2)
    assert outcome.steps[1].procedure_effect == "state_setting"
    assert outcome.steps[2].procedure_effect == "commit"
    assert outcome.steps[2].captures_completion_conditions == (0,)
    assert outcome.source_step_ids == ("step_1", "step_2", "step_3")
    assert outcome.steps[0].target_grounding == _evidence().steps[0].target_grounding


def test_compiler_removes_run_local_element_references() -> None:
    evidence = _evidence()
    expression = evidence.steps[0].proposal.target_expression
    first = evidence.steps[0].model_copy(
        update={
            "proposal": evidence.steps[0].proposal.model_copy(
                update={"target_element_ref": "discovery-element-1"}
            )
        }
    )
    evidence = evidence.model_copy(update={"steps": (first, *evidence.steps[1:])})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.steps[0].proposal.target_element_ref is None
    assert outcome.steps[0].proposal.target_expression == expression


def test_compiler_uses_next_selected_source_as_missing_expectation() -> None:
    evidence = _evidence()
    first = evidence.steps[0].model_copy(update={"destination_state_id": None})
    evidence = evidence.model_copy(update={"steps": (first, *evidence.steps[1:])})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.steps[0].expects.destination_state_id == evidence.steps[1].source_state_id


def test_compiled_source_dispatches_once_then_checks_live_expectation() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(
        evidence,
        _response(),
        identifier_factory=lambda: "procedure_1",
    )
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(
        procedure,
        evidence,
        identifier_factory=iter(("live_action", "next_action")).__next__,
    )
    before, after = _observations()[0], _observations()[2]
    request = PlanningRequest(
        observation=before,
        objective="Replay.",
        completion_conditions=("Compiled expectation passes.",),
        available_actions=frozenset({ActionKind.TAP}),
    )

    action = source.plan(request)
    outcome = (
        agent_result(
            before,
            after,
            action=ActionKind.TAP,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        .steps[0]
        .outcome
    )
    assert outcome is not None
    next_action = source.plan(
        request.model_copy(
            update={
                "observation": after,
                "history": (PlanningHistoryEntry(proposal=action, outcome=outcome),),
            }
        )
    )

    assert action.proposal_id == "live_action"
    assert source.completed_step_count == 1
    assert next_action.proposal_id == "next_action"


def test_compiled_source_stops_before_action_when_source_guard_diverges() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)

    terminal = source.plan(
        PlanningRequest(
            observation=_observations()[-1],
            objective="Replay.",
            completion_conditions=("Compiled expectation passes.",),
            available_actions=frozenset({ActionKind.TAP}),
        )
    )

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.divergence is not None
    assert source.divergence.guard == "source"


def test_procedure_grounder_reuses_exact_stored_pixels_before_fallback() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)
    source.plan(
        PlanningRequest(
            observation=_observations()[0],
            objective="Replay.",
            completion_conditions=("Compiled expectation passes.",),
            available_actions=frozenset({ActionKind.TAP}),
        )
    )
    stored = procedure.steps[0].target_grounding
    assert stored is not None

    class RefusingFallback:
        def required_model_requests(self, request):
            del request
            return 1

        def ground(self, request):
            raise AssertionError(f"Unexpected fallback for {request.grounding_id}")

    source_observation = _observations()[0]
    request = stored.request.model_copy(
        update={
            "grounding_id": "live_grounding",
            "source_presentation": source_observation.presentation_key,
        }
    )
    result = ProcedureGrounder(RefusingFallback(), source).ground(request)

    assert result.resolution_source is GroundingResolutionSource.PROCEDURE
    assert isinstance(result.resolution, ProcedureGroundingResolution)
    assert result.resolution.source_step_id == procedure.steps[0].source_step_id
    assert result.location == stored.result.location


@pytest.mark.parametrize(
    ("satisfied", "terminal"),
    (
        (True, ProcedureReplayTerminal.COMPLETED),
        (False, ProcedureReplayTerminal.REPLAY_COMPLETE),
    ),
)
def test_completed_procedure_terminal_requires_satisfied_audit(
    satisfied: bool,
    terminal: ProcedureReplayTerminal,
) -> None:
    verification = GoalVerification(
        verification_id="verification_1",
        objective_id="objective_1",
        final_observation_id="obs_4",
        satisfied=satisfied,
        reason="The configured final audit conclusion.",
    )
    source = cast(
        CompiledProcedureActionSource,
        SimpleNamespace(replay_complete=True),
    )

    result = _procedure_terminal(
        AgentTerminalReason.DONE_REQUESTED,
        source,
        None,
        None,
        verification,
    )

    assert result is terminal


@pytest.mark.parametrize(
    ("response", "check"),
    (
        (_response((1, 1, 3)), CompilerCheck.SELECTION_ONLY),
        (_response((1, 3, 2)), CompilerCheck.SELECTION_ONLY),
        (_response((1, 2, 99)), CompilerCheck.SELECTION_ONLY),
        (_response((2, 3)), CompilerCheck.STARTS_AT_LAUNCH),
        (_response((1, 3)), CompilerCheck.NO_LOST_STATE),
        (_response(milestones={}), CompilerCheck.COVERS_TASK),
    ),
)
def test_compiler_refuses_each_invalid_selection(
    response: ReconcilerResponse,
    check: CompilerCheck,
) -> None:
    outcome = compile_replay_procedure(_evidence(), response)

    assert isinstance(outcome, CompilerRefusal)
    assert check in _checks(outcome)


@pytest.mark.parametrize(
    ("result", "kind"),
    (
        (EvidenceResult.REFUSED, EvidenceStepKind.PATH),
        (EvidenceResult.CRASHED, EvidenceStepKind.PATH),
        (EvidenceResult.NO_EFFECT, EvidenceStepKind.PATH),
    ),
)
def test_compiler_refuses_bad_results_and_environment_steps(
    result: EvidenceResult,
    kind: EvidenceStepKind,
) -> None:
    evidence = _evidence()
    changed = evidence.steps[1].model_copy(update={"result": result, "kind": kind})
    evidence = evidence.model_copy(
        update={"steps": (evidence.steps[0], changed, evidence.steps[2])}
    )

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, CompilerRefusal)
    assert CompilerCheck.RESULTS_REUSABLE in _checks(outcome)


def test_ambiguous_result_compiles_with_a_warning() -> None:
    evidence = _evidence()
    changed = evidence.steps[1].model_copy(update={"result": EvidenceResult.AMBIGUOUS})
    evidence = evidence.model_copy(
        update={"steps": (evidence.steps[0], changed, evidence.steps[2])}
    )

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert any(warning.source_step == 2 for warning in outcome.warnings)


def test_selected_correction_can_replace_same_control() -> None:
    evidence = _evidence()
    named_control = evidence.steps[1].control.model_copy(update={"identity_key": "text_field:name"})
    original = evidence.steps[1].model_copy(update={"control": named_control})
    corrected_proposal = evidence.steps[2].proposal.model_copy(
        update={
            "action": evidence.steps[1].proposal.action,
            "target_expression": evidence.steps[1].proposal.target_expression,
            "text": evidence.steps[1].proposal.text,
        }
    )
    corrected = evidence.steps[2].model_copy(
        update={
            "proposal": corrected_proposal,
            "control": named_control,
        }
    )
    evidence = evidence.model_copy(update={"steps": (evidence.steps[0], original, corrected)})
    response = ReconcilerResponse(
        request_id="request_1",
        completion_conditions=evidence.task.completion_conditions,
        procedure=(1, 3),
        milestones={0: 3},
        decisions=(
            ReconcilerDecision(
                keep=(3,),
                drop=(2,),
                why="Step 3 is the corrected attempt for the same field.",
            ),
        ),
    )

    outcome = compile_replay_procedure(evidence, response)

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.source_step_ids == ("step_1", "step_3")


def test_correction_cannot_replace_work_after_dependent_selected_steps() -> None:
    evidence = _evidence()
    named_control = evidence.steps[1].control.model_copy(update={"identity_key": "text_field:name"})
    original = evidence.steps[1].model_copy(update={"control": named_control})
    corrected = evidence.steps[2].model_copy(
        update={
            "proposal": evidence.steps[2].proposal.model_copy(
                update={
                    "action": original.proposal.action,
                    "text": original.proposal.text,
                }
            ),
            "control": named_control,
            "step": 4,
        }
    )
    middle = evidence.steps[2].model_copy(update={"step": 3})
    evidence = evidence.model_copy(
        update={"steps": (evidence.steps[0], original, middle, corrected)}
    )
    response = ReconcilerResponse(
        request_id="request_1",
        completion_conditions=evidence.task.completion_conditions,
        procedure=(1, 3, 4),
        milestones={0: 4},
        decisions=(
            ReconcilerDecision(
                keep=(4,),
                drop=(2,),
                why="Step 4 repeats the field after other selected work.",
            ),
        ),
    )

    outcome = compile_replay_procedure(evidence, response)

    assert isinstance(outcome, CompilerRefusal)
    assert CompilerCheck.NO_LOST_STATE in _checks(outcome)


def test_refusal_persistence_never_creates_a_procedure(tmp_path: Path) -> None:
    outcome = compile_replay_procedure(_evidence(), _response((1, 3)))
    assert isinstance(outcome, CompilerRefusal)
    store = ReconciliationArtifactStore(tmp_path)

    path = store.persist_compilation(outcome)

    assert path == COMPILER_REFUSAL_PATH
    assert (tmp_path / COMPILER_REFUSAL_PATH).is_file()
    assert not (tmp_path / REPLAY_PROCEDURE_PATH).exists()


def test_compiled_procedure_store_loads_only_exact_compatible_objective(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path)

    saved = store.save(compatibility, evidence, procedure)

    assert (
        store.load(
            compatibility,
            evidence.task.objective_id,
            evidence.task.objective,
        )
        == saved
    )
    assert store.load(compatibility, evidence.task.objective_id, "Another objective.") is None


def test_compiled_procedure_store_lists_only_compatible_objective_candidates(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    other_evidence = evidence.model_copy(
        update={
            "task": evidence.task.model_copy(
                update={
                    "objective_id": "objective_2",
                    "objective": "Save another named item.",
                }
            )
        }
    )
    other_procedure = procedure.model_copy(update={"procedure_id": "procedure_2"})
    store = CompiledProcedureStore(tmp_path)
    store.save(compatibility, evidence, procedure)
    store.save(compatibility, other_evidence, other_procedure)

    candidates = store.compatible_candidates(compatibility)

    assert {item.procedure.procedure_id for item in candidates} == {
        procedure.procedure_id,
        "procedure_2",
    }
