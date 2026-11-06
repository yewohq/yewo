"""Exact-state requirements for deterministic procedure repair."""

import pytest

from yewo.agent.actuation import ActionKind
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure.records import ProcedureRepairTarget
from yewo.agent.replay.procedure.repair import _target_facts
from yewo.agent.replay.procedure.repair_targets import build_procedure_repair_preparation

from agent.replay.support import (
    evidence as _evidence,
    observations as _observations,
    response as _response,
)


@pytest.mark.parametrize("action", (ActionKind.TYPE, ActionKind.CLEAR))
def test_text_mutation_repair_requires_exact_content(action: ActionKind) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    step = procedure.steps[1].model_copy(
        update={
            "proposal": procedure.steps[1].proposal.model_copy(
                update={"action": action, "text": None}
            )
        }
    )

    preparation = build_procedure_repair_preparation(
        procedure,
        evidence,
        step,
        "expectation",
        step.expects.destination_state_id,
        "The field content diverged.",
    )

    assert preparation is not None
    assert preparation.requires_exact_content_match


def test_repair_context_bounds_remaining_step_summaries() -> None:
    evidence = _evidence()
    compiled = compile_replay_procedure(evidence, _response())
    assert isinstance(compiled, ReplayProcedure)
    template = compiled.steps[0]
    steps = tuple(
        template.model_copy(
            update={
                "procedure_index": index,
                "source_step": index + 1,
                "source_step_id": f"step-{index}",
            }
        )
        for index in range(7)
    )
    procedure = ReplayProcedure.model_validate(
        compiled.model_dump(mode="python")
        | {
            "source_step_ids": tuple(step.source_step_id for step in steps),
            "steps": steps,
        }
    )

    preparation = build_procedure_repair_preparation(
        procedure,
        evidence,
        steps[0],
        "source",
        steps[0].source_state_id,
        "The source state diverged.",
    )

    assert preparation is not None
    assert (
        preparation.target.reference_screenshot
        == evidence.states[steps[0].source_state_id].representative_screenshot
    )
    assert preparation.target.remaining_procedure_step_count == 7
    assert tuple(
        summary.procedure_index for summary in preparation.target.remaining_step_summaries
    ) == (0, 1, 2, 3, 4)


def test_keyboard_target_explains_focus_reset() -> None:
    target = ProcedureRepairTarget(
        expected_state_id="state_password",
        resume_procedure_index=4,
        presentation=_observations()[0].presentation_key.model_copy(
            update={"keyboard_visible": True}
        ),
        established_by_action=ActionKind.FOCUS,
        established_by_target_expression="the Password text field",
        reason="The keyboard closed after password focus.",
    )

    assert any("focus one other visible text field once" in fact for fact in _target_facts(target))
