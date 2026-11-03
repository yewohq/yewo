"""Per-record AppMemory capture skip tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.memory.knowledge_capture import build_memory_delta

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


def test_negative_transition_duration_skips_only_timing(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000", offset_seconds=2)
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        datetime(2026, 11, 3, tzinfo=timezone.utc),
    )

    assert delta.timing == ()
    assert len(delta.skipped_records) == 1
    assert delta.skipped_records[0].record_kind == "timing"
