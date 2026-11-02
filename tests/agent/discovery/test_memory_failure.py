"""Discovery evidence retained when app-memory persistence fails."""

from pathlib import Path
from typing import cast

import pytest

from yewo.agent import AppMemoryStoreError
from yewo.agent.discovery import DiscoveryObjectiveState, DiscoveryResult
from yewo.models import ModelClient
from yewo.runs import NeverInterrupted

from agent.discovery.support import agent_result, observation
from agent.discovery.test_workflow import StubAgent, StubMemoryStore, _context, _workflow


class FailingMemoryStore(StubMemoryStore):
    def save(self, memory, source_run_root):
        del memory, source_run_root
        raise AppMemoryStoreError("The test memory write failed.")


def test_memory_write_failure_persists_unresolved_discovery_evidence(
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
    workflow = _workflow(tmp_path, FailingMemoryStore())
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

    with pytest.raises(AppMemoryStoreError):
        workflow.execute(_context(tmp_path), NeverInterrupted())

    result_path = tmp_path / "run/assets/agent/discovery/result.json"
    result = DiscoveryResult.model_validate_json(result_path.read_text())
    assert result.memory_persistence_error == "Yewo could not update app memory."
    assert result.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED
    assert result.agent_result.steps
