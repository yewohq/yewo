"""Two-tier narration for discovery conclusions."""

from pathlib import Path
from typing import cast

import pytest

from yewo.agent import AgentTerminalReason
from yewo.agent.discovery.narration import narrate_replay_handoff
from yewo.agent.route_replay import ReplayPathKind, RouteReplayResult, RouteReplayTerminal
from yewo.models import ModelClient
from yewo.run_narration import RunNarrator, narration_scope
from yewo.runs import NeverInterrupted

from agent.discovery.support import agent_result, observation
from agent.discovery.test_workflow import (
    StubAgent,
    StubMemoryStore,
    _context,
    _workflow,
)


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
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    with narration_scope(RunNarrator(tmp_path)):
        workflow.execute(_context(tmp_path), NeverInterrupted())

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Starting discovery for 1 objective(s)." in narration
    assert "No compatible verified route is available. Exploring the app." in narration
    assert "Discovery finished with 1 of 1 objective(s) supported." in narration
    assert "Independent goal verification satisfied" in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "request=request_1 objectives=1" in trace
    assert "flow_lookup=miss reason=no_compatible_verified_route_or_checkpoint" in trace
    assert "objective=account_form state=supported" in trace
    assert "memory=" in trace


def test_discovery_narrates_invalidated_stored_flow(tmp_path: Path) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    replay = RouteReplayResult(
        path_id="checkpoint_1",
        path_kind=ReplayPathKind.DISCOVERY_CHECKPOINT,
        terminal=RouteReplayTerminal.DIVERGED,
        completed_segment_count=2,
        failed_segment_index=2,
        reason="The source guard changed.",
        agent_result=agent_result(
            before,
            after,
            terminal=AgentTerminalReason.CANNOT_PROGRESS,
        ),
    )

    with narration_scope(RunNarrator(tmp_path)):
        narrate_replay_handoff(replay)

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Stored flow was invalidated at segment 2." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert (
        "flow=checkpoint_1 flow_kind=discovery_checkpoint "
        "reuse=invalidated failed_segment=2"
    ) in trace
