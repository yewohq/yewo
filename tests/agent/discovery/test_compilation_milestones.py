"""Milestone evidence derived from provisional discovery compilation."""

from pathlib import Path
from typing import cast

import pytest

from yewo.observations import ScreenComparisonPolicy
from yewo.agent import AgentTerminalReason
from yewo.agent.discovery.compilation_milestones import (
    DiscoveryCompilationMilestones,
    compile_completed_discovery,
)
from yewo.agent.replay import (
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
    compile_replay_procedure,
)
from yewo.agent.replay.reconciliation.store import (
    EVIDENCE_DOCUMENT_PATH,
    RECONCILER_RESPONSE_PATH,
)
from yewo.models import ModelCallDisposition, ModelClient

from yewo.agent.discovery.records import DiscoveryResult

from agent.discovery.support import observation
from agent.discovery.test_procedure_workflow import _procedure_replay
from agent.discovery.test_workflow import _discovery_request
from agent.replay.support import evidence as _evidence, response as _response


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


def test_repaired_replay_prefix_is_included_in_provisional_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yewo.agent.discovery import compilation_milestones as module

    before = observation("replay_before", "0000000000000000")
    after = observation(
        "replay_after",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    replay = _procedure_replay(before, after, satisfied=False)
    repair_result = replay.agent_result.model_copy(
        update={"terminal_reason": AgentTerminalReason.DONE_REQUESTED}
    )
    captured: dict[str, DiscoveryResult] = {}

    def compile_milestones(**kwargs):
        captured["result"] = kwargs["result"]
        return cast(DiscoveryCompilationMilestones, object())

    monkeypatch.setattr(module, "compile_discovery_milestones", compile_milestones)

    compilation = compile_completed_discovery(
        replay,
        tmp_path,
        "run_1",
        _discovery_request(),
        repair_result,
        cast(ModelClient, object()),
        "replay-preparer",
        ScreenComparisonPolicy(),
    )

    assert compilation is not None
    assert captured["result"].procedure_replay is replay
    assert captured["result"].agent_result is repair_result


def test_valid_milestone_survives_unrelated_procedure_refusal() -> None:
    evidence = _evidence()
    response = _response(procedure=(1, 2, 3, 99), milestones={0: 3})
    report = ProcedureCompilationReport(
        source_run_id="run_1",
        evidence_id=evidence.evidence_id,
        request_id="request_1",
        status=ProcedureCompilationStatus.COMPILER_REFUSED,
        model_disposition=ModelCallDisposition.COMPLETED,
        reason="The procedure selection was refused.",
        compiler_refusal_reason_count=1,
    )
    compilation = DiscoveryCompilationMilestones(
        report=report,
        evidence=evidence,
        response=response,
    )
    source = observation("obs_3", "bbbbbbbbbbbbbbbb")
    captured = observation("obs_4", "aaaaaaaaaaaaaaaa")

    milestones = compilation.goal_milestones({"obs_3": source, "obs_4": captured})

    assert compilation.observation_ids() == {"obs_3", "obs_4"}
    assert len(milestones) == 1
    assert milestones[0].completion_condition_index == 0
    assert tuple(action.procedure_index for action in milestones[0].supporting_actions) == (0, 1, 2)


def test_unselected_current_run_milestone_remains_audit_evidence() -> None:
    evidence = _evidence()
    response = _response(procedure=(1, 2), milestones={0: 3})
    report = ProcedureCompilationReport(
        source_run_id="run_1",
        evidence_id=evidence.evidence_id,
        request_id="request_1",
        status=ProcedureCompilationStatus.COMPILER_REFUSED,
        model_disposition=ModelCallDisposition.COMPLETED,
        reason="The procedure omitted one non-reusable milestone action.",
        compiler_refusal_reason_count=1,
    )
    compilation = DiscoveryCompilationMilestones(
        report=report,
        evidence=evidence,
        response=response,
    )
    source = observation("obs_3", "bbbbbbbbbbbbbbbb")
    captured = observation("obs_4", "aaaaaaaaaaaaaaaa")

    milestones = compilation.goal_milestones({"obs_3": source, "obs_4": captured})

    assert len(milestones) == 1
    assert tuple(action.step_index for action in milestones[0].supporting_actions) == (1, 2, 3)
    assert tuple(action.procedure_index for action in milestones[0].supporting_actions) == (
        0,
        1,
        None,
    )


def test_a_refused_compilation_retains_valid_audit_milestones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yewo.agent.discovery import compilation_milestones as module

    evidence = _evidence()
    response = _response(procedure=(1, 2, 3, 99), milestones={0: 3})
    refused = ProcedureCompilationReport(
        source_run_id="run_1",
        evidence_id=evidence.evidence_id,
        request_id="request_1",
        status=ProcedureCompilationStatus.COMPILER_REFUSED,
        model_disposition=ModelCallDisposition.COMPLETED,
        compiler_refusal_reason_count=1,
        reason="The compiler refused a non-reusable step.",
    )
    monkeypatch.setattr(module, "compile_completed_discovery_run", lambda **kwargs: refused)
    evidence_path = tmp_path / EVIDENCE_DOCUMENT_PATH
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(evidence.model_dump_json())
    response_path = tmp_path / RECONCILER_RESPONSE_PATH
    response_path.write_text(response.model_dump_json())

    compilation = module.compile_discovery_milestones(
        root=tmp_path,
        run_id="run_1",
        result=cast(DiscoveryResult, object()),
        model_client=cast(ModelClient, object()),
        model_profile_id="reconciler-primary",
        comparison=ScreenComparisonPolicy(),
    )

    assert compilation.procedure is None
    assert compilation.evidence == evidence
    assert compilation.response == response
    source = observation("obs_3", "bbbbbbbbbbbbbbbb")
    captured = observation("obs_4", "aaaaaaaaaaaaaaaa")
    milestones = compilation.goal_milestones({"obs_3": source, "obs_4": captured})
    assert len(milestones) == 1
    assert milestones[0].completion_condition_index == 0
