"""Procedure-specific AppMemory finding tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay.procedure.records import ProcedureReplayResult, ProcedureReplayTerminal

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


def test_verified_shorter_run_flags_procedure_optimization_not_failure(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    evidence = agent_result(before, after)
    replay = ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="evidence_1",
        source_run_id="run_1",
        setup_id="attempt_1",
        application=memory.compatibility.application,
        terminal=ProcedureReplayTerminal.BUDGET_EXHAUSTED,
        procedure_step_count=2,
        completed_step_count=1,
        reason="Discovery proved the task before the retained suffix was needed.",
        agent_result=evidence,
    )
    verification = GoalVerification(
        verification_id="verification_1",
        objective_id="account_form",
        final_observation_id=after.observation_id,
        satisfied=True,
        reason="The current run proves the objective.",
        observed_app_state="The account form is visible.",
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        evidence,
        {before.observation_id: before, after.observation_id: after},
        datetime(2026, 11, 3, tzinfo=timezone.utc),
        verification=verification,
        procedure_replay=replay,
    )

    procedure_findings = [
        finding for finding in delta.findings if "procedure" in finding.text.casefold()
    ]
    assert len(procedure_findings) == 1
    assert "verified after fewer" in procedure_findings[0].text
