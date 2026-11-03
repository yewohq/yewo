"""AppMemory timing policy for compiled procedure settlement."""

from pathlib import Path

import pytest

from yewo.agent.memory.knowledge import (
    AppKnowledge,
    KnowledgeEvidenceReference,
    TransitionTimingKnowledge,
    TransitionTimingSample,
)
from yewo.agent.memory.records import AppMemory
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.replay.procedure_settlement import ProcedureSettlementPolicySource
from yewo.agent.replay.reconciliation_compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation_records import ReplayProcedure
from yewo.agent.runtime.observation_settlement import ActionSettlementPolicy

from agent.memory.test_store import _memory
from agent.replay.test_procedure_compiler import _evidence, _response


def _procedure() -> ReplayProcedure:
    procedure = compile_replay_procedure(_evidence(), _response())
    assert isinstance(procedure, ReplayProcedure)
    return procedure


def _decision(procedure: ReplayProcedure) -> PlannerDecisionMetadata:
    step = procedure.steps[0]
    return PlannerDecisionMetadata(
        source=PlannerDecisionSource.COMPILED_PROCEDURE,
        reused_flow_id=procedure.procedure_id,
        reused_procedure_step_ids=(step.source_step_id,),
        reused_segment_index=step.procedure_index,
    )


def _timed_memory(tmp_path: Path, measurement: str) -> AppMemory:
    unused_compatibility, memory = _memory(tmp_path)
    transition = next(item for item in memory.transitions if item.source_step_id == "step_1")
    samples = tuple(
        TransitionTimingSample.model_validate(
            {
                "sample_id": f"timing_{index}",
                "measurement": measurement,
                "duration_ms": duration_ms,
                "source": KnowledgeEvidenceReference(run_id="run_1"),
            }
        )
        for index, duration_ms in enumerate((600, 700, 800), start=1)
    )
    timing = TransitionTimingKnowledge(
        transition_id=transition.transition_id,
        samples=samples,
    )
    return memory.model_copy(update={"app_knowledge": AppKnowledge(timing=(timing,))})


def test_repeated_dispatch_timing_shortens_compiled_action_polling(
    tmp_path: Path,
) -> None:
    procedure = _procedure()
    memory = _timed_memory(tmp_path, "dispatch_to_settled_v2")
    source = ProcedureSettlementPolicySource(procedure, memory)
    default = ActionSettlementPolicy(
        maximum_observations=6,
        initial_delay_seconds=0.35,
        interval_seconds=0.35,
    )

    resolved = source.resolve(default, _decision(procedure))

    assert resolved.initial_delay_seconds == 0
    assert resolved.interval_seconds == pytest.approx(0.24)
    assert resolved.maximum_observations == 6
    assert resolved.required_stable_observations == 1


def test_legacy_timing_uses_fast_replay_policy_without_tuning(
    tmp_path: Path,
) -> None:
    procedure = _procedure()
    memory = _timed_memory(tmp_path, "source_to_settled_v1")
    source = ProcedureSettlementPolicySource(procedure, memory)
    default = ActionSettlementPolicy(
        maximum_observations=6,
        initial_delay_seconds=0.35,
        interval_seconds=0.35,
    )

    resolved = source.resolve(default, _decision(procedure))

    assert resolved.initial_delay_seconds == 0
    assert resolved.interval_seconds == pytest.approx(0.42)
    assert resolved.maximum_observations == 6
    assert resolved.required_stable_observations == 1


def test_fast_replay_policy_preserves_the_configured_maximum_wait_window() -> None:
    procedure = _procedure()
    source = ProcedureSettlementPolicySource(procedure, None)
    default = ActionSettlementPolicy(
        maximum_observations=6,
        initial_delay_seconds=0.35,
        interval_seconds=0.35,
    )

    resolved = source.resolve(default, _decision(procedure))

    default_window = default.initial_delay_seconds + default.interval_seconds * 5
    resolved_window = resolved.initial_delay_seconds + resolved.interval_seconds * 5
    assert resolved_window == pytest.approx(default_window)
    assert resolved.required_stable_observations == 1


def test_timing_does_not_change_model_repair_settlement(tmp_path: Path) -> None:
    procedure = _procedure()
    memory = _timed_memory(tmp_path, "dispatch_to_settled_v2")
    source = ProcedureSettlementPolicySource(procedure, memory)
    default = ActionSettlementPolicy(
        maximum_observations=6,
        initial_delay_seconds=0.35,
        interval_seconds=0.35,
    )

    resolved = source.resolve(
        default,
        PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL),
    )

    assert resolved == default
    assert resolved.required_stable_observations == 2
