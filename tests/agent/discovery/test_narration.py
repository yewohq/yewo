"""Two-tier narration for discovery conclusions."""

from pathlib import Path
from typing import cast

import pytest

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
    assert "stored_route=none" in trace
    assert "objective=account_form state=supported" in trace
    assert "memory=" in trace
