"""Regressions found by live compiled-procedure runs."""

import pytest

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionOutcomeSignal
from yewo.agent.replay import (
    EvidenceStepKind,
    ProcedureEffect,
    ReplayProcedure,
    build_evidence_document,
    compile_replay_procedure,
)
from yewo.agent.screen_analysis import (
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
    ScreenWorkflowRole,
)
from yewo.agent.replay.reconciliation.compiler import procedure_effect

from agent.discovery.support import agent_result, observation
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource

from agent.replay.support import (
    evidence as _evidence,
    observations as _observations,
    response as _response,
)
from agent.replay.test_procedure_repair import (
    _evidence as _procedure_evidence,
    _outcome,
    _request,
)


def test_compiler_does_not_require_visible_text_signal_for_masked_input() -> None:
    evidence = _evidence()
    focus = evidence.steps[0].model_copy(
        update={
            "action": ActionKind.FOCUS,
            "proposal": evidence.steps[0].proposal.model_copy(
                update={
                    "action": ActionKind.FOCUS,
                    "target_expression": "the Password field",
                }
            ),
        }
    )
    typed = evidence.steps[1].model_copy(
        update={
            "signals": frozenset(
                {
                    ActionOutcomeSignal.PIXELS_CHANGED,
                    ActionOutcomeSignal.TEXT_LANDED,
                }
            )
        }
    )
    evidence = evidence.model_copy(update={"steps": (focus, typed, evidence.steps[2])})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.steps[1].expects.required_signals == frozenset(
        {ActionOutcomeSignal.PIXELS_CHANGED}
    )


@pytest.mark.parametrize("action", (ActionKind.TYPE, ActionKind.CLEAR))
def test_compiler_ignores_incidental_focus_change(action: ActionKind) -> None:
    evidence = _evidence()
    proposal_update: dict[str, object] = {"action": action}
    if action is ActionKind.CLEAR:
        proposal_update["text"] = None
    typed = evidence.steps[1].model_copy(
        update={
            "action": action,
            "proposal": evidence.steps[1].proposal.model_copy(update=proposal_update),
            "signals": frozenset(
                {
                    ActionOutcomeSignal.PIXELS_CHANGED,
                    ActionOutcomeSignal.FOCUS_CHANGED,
                }
            ),
        }
    )
    evidence = evidence.model_copy(update={"steps": (evidence.steps[0], typed, evidence.steps[2])})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.steps[1].expects.required_signals == frozenset(
        {ActionOutcomeSignal.PIXELS_CHANGED}
    )


def test_dismiss_keyboard_is_application_path_work() -> None:
    before = observation("keyboard_open", "1111111111111111").model_copy(
        update={"keyboard_visible": True}
    )
    after = observation(
        "keyboard_closed",
        "2222222222222222",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    ).model_copy(update={"keyboard_visible": False})
    step = agent_result(
        before,
        after,
        action=ActionKind.DISMISS_KEYBOARD,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    evidence = build_evidence_document(
        run_id="run_keyboard",
        objective_id="objective_keyboard",
        task=AgentTask(
            objective="Submit the form.",
            completion_conditions=("The form is submitted.",),
            discovery_mode=True,
        ),
        steps=(step,),
        observations=(before, after),
        identifier_factory=lambda: "evidence_keyboard",
    )

    assert evidence.steps[0].kind is EvidenceStepKind.PATH


def test_focus_and_keyboard_dismissal_are_navigation_effects() -> None:
    focus = _evidence().steps[0]
    focus = focus.model_copy(
        update={"proposal": focus.proposal.model_copy(update={"action": ActionKind.FOCUS})}
    )
    dismissal = focus.model_copy(
        update={
            "proposal": focus.proposal.model_copy(update={"action": ActionKind.DISMISS_KEYBOARD})
        }
    )

    assert procedure_effect(focus) is ProcedureEffect.NAVIGATION
    assert procedure_effect(dismissal) is ProcedureEffect.NAVIGATION


def test_analyzed_advance_stage_is_a_commit_effect() -> None:
    step = _evidence().steps[0]
    step = step.model_copy(
        update={
            "control": step.control.model_copy(
                update={"workflow_role": ScreenWorkflowRole.ADVANCE_STAGE}
            ),
        }
    )

    assert procedure_effect(step) is ProcedureEffect.COMMIT


def test_link_semantic_role_produces_a_navigation_effect() -> None:
    step = _evidence().steps[0]
    step = step.model_copy(
        update={
            "control": step.control.model_copy(
                update={
                    "kind": ScreenElementKind(
                        visual=ScreenVisualKind.TEXT,
                        semantic=ScreenSemanticRole.LINK,
                    )
                }
            ),
        }
    )

    assert procedure_effect(step) is ProcedureEffect.NAVIGATION


def test_a_stale_repair_target_cannot_rewind_the_procedure_cursor() -> None:
    evidence = _procedure_evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)
    screens = _observations()
    first = source.plan(_request(screens[0]))
    source.plan(_request(screens[-1], (_outcome(first, screens[0], screens[-1]),)))

    assert source.rejoin(screens[2]) is True
    second = source.plan(_request(screens[2]))
    source.plan(_request(screens[3], (_outcome(second, screens[2], screens[3]),)))
    assert source.next_procedure_index == 2

    # A target kept past its rejoin would resume at its own index, so a later
    # match rewinds the cursor and redispatches work the run already completed.
    assert source.rejoin(screens[2]) is False
    assert source.next_procedure_index == 2
