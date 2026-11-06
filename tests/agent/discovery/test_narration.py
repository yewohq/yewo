"""Two-tier narration for discovery conclusions."""

from pathlib import Path
from typing import cast

import pytest

from yewo.agent.discovery.narration import narrate_discovery_result
from yewo.agent.discovery.narration import narrate_compiled_procedure_selection
from yewo.agent.discovery.coverage import procedure_replay_coverage
from yewo.agent.discovery.records import DiscoveryResult
from yewo.models import ModelClient
from yewo.run_narration import RunNarrator, narration_scope
from yewo.runs import NeverInterrupted

from agent.discovery.support import agent_result, observation
from agent.discovery.test_workflow import (
    StubAgent,
    StubMemoryStore,
    _context,
    _discovery_request,
    _stub_compilation,
    _workflow,
)
from agent.discovery.test_procedure_workflow import _procedure_replay


def test_discovery_narrates_coverage_verification_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    agent = StubAgent(agent_result(before, after))
    workflow = _workflow(tmp_path, StubMemoryStore())
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.compile_completed_discovery",
        lambda *args, **kwargs: _stub_compilation(),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    with narration_scope(RunNarrator(tmp_path)):
        workflow.execute(_context(tmp_path), NeverInterrupted())

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Starting discovery for 1 objective(s)." in narration
    assert "No compatible compiled procedure is available. Exploring the app." in narration
    assert "Discovery compiled a reusable procedure." in narration
    assert "Reusable objective coverage is separate: 1 of 1 objective(s) supported." in narration
    assert "Independent goal verification satisfied" in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "request=request_1 objectives=1" in trace
    assert "flow_lookup=miss reason=no_compatible_compiled_procedure" in trace
    assert "objective=record_form coverage_state=supported" in trace
    assert "memory=" in trace


def test_discovery_narrates_compiled_procedure_selection(tmp_path: Path) -> None:
    with narration_scope(RunNarrator(tmp_path)):
        narrate_compiled_procedure_selection("procedure_1", 67)

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Replaying a compatible compiled procedure." in narration
    assert "No compatible compiled procedure" not in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "flow_kind=compiled_procedure flow=procedure_1 steps=67" in trace


def test_procedure_replay_narration_separates_task_and_objective_coverage(
    tmp_path: Path,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    request = _discovery_request()
    replay = _procedure_replay(before, after, satisfied=True)
    result = DiscoveryResult(
        request=request,
        terminal_reason=replay.agent_result.terminal_reason,
        coverage=procedure_replay_coverage(request, replay),
        agent_result=replay.agent_result,
        procedure_replay=replay,
        verification=replay.verification,
        memory_id="memory_1",
    )

    with narration_scope(RunNarrator(tmp_path)):
        narrate_discovery_result(result)

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Compiled procedure replay finished as completed." in narration
    assert "Task verification was satisfied." in narration
    assert "Compiled-procedure reuse is available." in narration
    assert "Reusable objective coverage is separate: 0 of 1" in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "task_verification=satisfied" in trace
    assert "procedure_replay=completed" in trace
    assert "compiled_procedure_reusable=true" in trace
    assert "objectives_supported=0" in trace
    assert "objective=record_form coverage_state=unresolved" in trace


def test_compiled_discovery_narration_does_not_call_procedure_reuse_unresolved(
    tmp_path: Path,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    request = _discovery_request()
    replay = _procedure_replay(before, after, satisfied=True)
    result = DiscoveryResult(
        request=request,
        terminal_reason=replay.agent_result.terminal_reason,
        coverage=procedure_replay_coverage(request, replay),
        agent_result=replay.agent_result,
        verification=replay.verification,
        compiled_procedure_id="procedure_1",
        compiled_procedure_step_count=8,
        memory_id="memory_1",
    )

    with narration_scope(RunNarrator(tmp_path)):
        narrate_discovery_result(result)

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Discovery compiled a reusable procedure." in narration
    assert "Reusable objective coverage is separate: 0 of 1" in narration
    assert "Discovery finished with 0 of 1" not in narration
