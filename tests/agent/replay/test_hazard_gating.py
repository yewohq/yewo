"""Deterministic AppMemory hazard gates for compiled procedures."""

from pathlib import Path

from yewo.agent.actuation import ActionKind
from yewo.agent.memory.knowledge import (
    AppKnowledge,
    HazardKind,
    HazardKnowledge,
    KnowledgeEvidenceReference,
    KnowledgeRule,
)
from yewo.agent.memory.records import AppMemory
from yewo.agent.replay import CompilerCheck, CompilerRefusal, ReplayProcedure
from yewo.agent.replay.procedure.hazards import compiled_procedure_hazard_reasons
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure

from agent.memory.test_store import _memory
from agent.replay.test_procedure_compiler import _evidence, _response


def memory_with_hazard(tmp_path: Path, *, target: str | None = None) -> AppMemory:
    """Build valid memory with one platform-observed source-screen hazard."""
    unused_compatibility, memory = _memory(tmp_path)
    source_screen = next(
        screen
        for screen in memory.screens
        if any(variant.observation_id == "obs_1" for variant in screen.variants)
    )
    hazard = HazardKnowledge(
        hazard_id="hazard_1",
        screen_id=source_screen.screen_id,
        action=ActionKind.TAP,
        target_description=target,
        kind=HazardKind.CRASH,
        seen_count=1,
        sources=(
            KnowledgeEvidenceReference(
                run_id="run_1",
                step_id="step_1",
                observation_id="obs_1",
            ),
        ),
    )
    return memory.model_copy(update={"app_knowledge": AppKnowledge(hazards=(hazard,))})


def test_compiler_refuses_platform_observed_hazard_with_exact_sources(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    memory = memory_with_hazard(
        tmp_path,
        target=evidence.steps[0].proposal.target_expression,
    )

    outcome = compile_replay_procedure(evidence, _response(), memory=memory)

    assert isinstance(outcome, CompilerRefusal)
    reason = next(
        item for item in outcome.reasons if item.check is CompilerCheck.AVOIDS_KNOWN_HAZARDS
    )
    assert reason.source_step == 1
    assert reason.hazard_id == "hazard_1"
    assert reason.hazard_sources[0].run_id == "run_1"
    assert reason.hazard_sources[0].step_id == "step_1"


def test_compiler_does_not_use_model_authored_rule_as_a_safety_gate(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    rule = KnowledgeRule(
        rule_id="rule_1",
        text="Do not submit until every required field is complete.",
        sources=(KnowledgeEvidenceReference(run_id="run_1"),),
    )
    memory = memory.model_copy(update={"app_knowledge": AppKnowledge(rules=(rule,))})

    outcome = compile_replay_procedure(_evidence(), _response(), memory=memory)

    assert isinstance(outcome, ReplayProcedure)


def test_stored_procedure_is_rechecked_after_memory_learns_a_hazard(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    memory = memory_with_hazard(
        tmp_path,
        target=evidence.steps[0].proposal.target_expression,
    )

    reasons = compiled_procedure_hazard_reasons(evidence, procedure, memory)

    assert len(reasons) == 1
    assert reasons[0].hazard_id == "hazard_1"
