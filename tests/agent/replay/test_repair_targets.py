"""Exact-state requirements for deterministic procedure repair."""

import pytest

from yewo.agent.actuation import ActionKind
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure.repair_targets import build_procedure_repair_preparation

from agent.replay.support import evidence as _evidence, response as _response


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
