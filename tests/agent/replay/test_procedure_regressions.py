"""Regressions found by live compiled-procedure runs."""

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionOutcomeSignal
from yewo.agent.replay import (
    ControlRole,
    EvidenceStepKind,
    ProcedureEffect,
    ReplayProcedure,
    build_evidence_document,
    compile_replay_procedure,
)
from yewo.agent.replay.reconciliation_compiler import procedure_effect

from agent.discovery.support import agent_result, observation
from agent.replay.test_procedure_compiler import _evidence, _response


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
    focus = _evidence().steps[0].model_copy(update={"action": ActionKind.FOCUS})
    dismissal = focus.model_copy(update={"action": ActionKind.DISMISS_KEYBOARD})

    assert procedure_effect(focus) is ProcedureEffect.NAVIGATION
    assert procedure_effect(dismissal) is ProcedureEffect.NAVIGATION


def test_generic_submit_target_is_a_commit_effect() -> None:
    step = _evidence().steps[0]
    step = step.model_copy(
        update={
            "proposal": step.proposal.model_copy(
                update={"target_expression": "the primary Submit button"}
            )
        }
    )

    assert procedure_effect(step) is ProcedureEffect.COMMIT


def test_navigation_role_wins_over_neighboring_commit_text() -> None:
    step = _evidence().steps[0]
    step = step.model_copy(
        update={
            "control": step.control.model_copy(update={"role": ControlRole.LINK}),
            "proposal": step.proposal.model_copy(
                update={
                    "target_expression": ("The Sign In link below the Register & Continue button.")
                }
            ),
        }
    )

    assert procedure_effect(step) is ProcedureEffect.NAVIGATION
