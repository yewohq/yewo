"""Milestone evidence derived from provisional discovery compilation."""

from yewo.agent.discovery.compilation_milestones import (
    DiscoveryCompilationMilestones,
)
from yewo.agent.replay import (
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
    compile_replay_procedure,
)
from yewo.models import ModelCallDisposition

from agent.discovery.support import observation
from agent.replay.test_procedure_compiler import _evidence, _response


def test_compiled_capture_becomes_goal_verification_milestone() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    report = ProcedureCompilationReport(
        source_run_id="run_1",
        evidence_id=evidence.evidence_id,
        request_id="request_1",
        status=ProcedureCompilationStatus.COMPILED,
        model_disposition=ModelCallDisposition.COMPLETED,
        procedure_id=procedure.procedure_id,
        reason="The procedure compiled.",
    )
    compilation = DiscoveryCompilationMilestones(
        report=report,
        evidence=evidence,
        procedure=procedure,
    )
    source = observation("obs_3", "bbbbbbbbbbbbbbbb")
    captured = observation("obs_4", "aaaaaaaaaaaaaaaa")

    assert compilation.observation_ids() == {"obs_3", "obs_4"}
    milestones = compilation.goal_milestones({"obs_3": source, "obs_4": captured})
    assert len(milestones) == 1
    assert milestones[0].completion_condition_index == 0
    assert milestones[0].source_observation is source
    assert milestones[0].observation is captured
    assert tuple(action.procedure_index for action in milestones[0].supporting_actions) == (0, 1, 2)
