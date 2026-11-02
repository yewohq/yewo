"""Structured replay proof and derived cross-run reporting tests."""

from pathlib import Path
import subprocess
import sys

from yewo.agent import AgentTerminalReason, PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.discovery import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.replay_evidence import (
    REPLAY_ATTEMPT_PATH,
    REPLAY_EVIDENCE_PATH,
    ReplayRunEvidence,
    build_replay_run_evidence,
    build_replay_usage_report,
    build_replay_usage_report_from_snapshot,
    load_replay_evidence_snapshot,
    persist_initial_replay_evidence,
    repeated_failed_transition,
)
from yewo.agent.route_replay import ReplayPathKind, RouteReplayResult, RouteReplayTerminal

from agent.discovery.support import agent_result, observation


def _discovery_result(replay: RouteReplayResult) -> DiscoveryResult:
    request = DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="objective_1",
                statement="The destination is reachable.",
            ),
        ),
    )
    return DiscoveryResult(
        request=request,
        terminal_reason=replay.agent_result.terminal_reason,
        coverage=DiscoveryCoverage(
            entries=(
                DiscoveryCoverageEntry(
                    objective_id="objective_1",
                    state=DiscoveryObjectiveState.UNRESOLVED,
                    reason="The replay result is retained without a capability claim.",
                ),
            )
        ),
        agent_result=replay.agent_result,
        replay=replay,
    )


def test_completed_replay_proof_resolves_to_steps_and_transitions(tmp_path: Path) -> None:
    before = observation("obs_1", "0" * 16)
    after = observation("obs_2", "1" * 16, predecessor_observation_id="obs_1")
    result = agent_result(before, after, terminal=AgentTerminalReason.DONE_REQUESTED)
    action_step, terminal_step = result.steps
    replay_result = result.model_copy(
        update={
            "steps": (
                action_step.model_copy(
                    update={
                        "decision": PlannerDecisionMetadata(
                            source=PlannerDecisionSource.VERIFIED_ROUTE,
                            reused_flow_id="route_1",
                            reused_transition_ids=("transition_1",),
                            reused_segment_index=0,
                        )
                    }
                ),
                terminal_step.model_copy(
                    update={
                        "decision": PlannerDecisionMetadata(
                            source=PlannerDecisionSource.VERIFIED_ROUTE,
                            reused_flow_id="route_1",
                        )
                    }
                ),
            )
        }
    )
    replay = RouteReplayResult(
        path_id="route_1",
        path_kind=ReplayPathKind.VERIFIED_ROUTE,
        terminal=RouteReplayTerminal.COMPLETED,
        completed_segment_count=1,
        replay_transition_ids=("transition_1",),
        reason="Every route guard matched.",
        agent_result=replay_result,
    )

    evidence = build_replay_run_evidence("run_1", _discovery_result(replay))
    run_root = tmp_path / "run_1"
    persist_initial_replay_evidence(run_root, "run_1", replay)
    persisted = ReplayRunEvidence.model_validate_json((run_root / REPLAY_ATTEMPT_PATH).read_text())
    report = build_replay_usage_report(tmp_path)

    assert evidence is not None
    assert evidence.full_replay_succeeded
    assert evidence.replay_step_ids == (action_step.step_id,)
    assert evidence.executed_transition_ids == ("transition_1",)
    assert persisted == evidence
    assert report.evidence_run_ids == ("run_1",)


def test_empty_completed_selection_is_not_counted_as_success() -> None:
    evidence = ReplayRunEvidence(
        run_id="run_1",
        path_id="checkpoint_1",
        path_kind=ReplayPathKind.DISCOVERY_CHECKPOINT,
        terminal=RouteReplayTerminal.COMPLETED,
        completed_segment_count=0,
    )

    assert not evidence.full_replay_succeeded


def test_replay_reporting_module_imports_without_discovery_import_ordering() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from yewo.agent.replay_evidence import build_replay_usage_report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_usage_report_counts_unique_attempts_completions_divergence_and_rejoin(
    tmp_path: Path,
) -> None:
    evidence = (
        ReplayRunEvidence(
            run_id="run_1",
            path_id="route_1",
            path_kind=ReplayPathKind.VERIFIED_ROUTE,
            terminal=RouteReplayTerminal.COMPLETED,
            completed_segment_count=2,
            replay_transition_ids=("transition_1", "transition_2"),
            executed_transition_ids=("transition_1", "transition_2"),
            replay_step_ids=("step_1", "step_2"),
        ),
        ReplayRunEvidence(
            run_id="run_2",
            path_id="route_1",
            path_kind=ReplayPathKind.VERIFIED_ROUTE,
            terminal=RouteReplayTerminal.DIVERGED,
            completed_segment_count=1,
            failed_segment_index=1,
            replay_transition_ids=("transition_1", "transition_2"),
            executed_transition_ids=("transition_1", "transition_2"),
            replay_step_ids=("step_3", "step_4"),
            rejoined_transition_ids=("transition_2",),
            rejoined_step_ids=("step_5",),
        ),
    )
    for item in evidence:
        path = tmp_path / item.run_id / REPLAY_EVIDENCE_PATH
        path.parent.mkdir(parents=True)
        path.write_text(item.model_dump_json(indent=2))
    attempt_path = tmp_path / "run_1" / REPLAY_ATTEMPT_PATH
    attempt_path.write_text(
        evidence[0]
        .model_copy(
            update={
                "terminal": RouteReplayTerminal.DIVERGED,
                "completed_segment_count": 0,
                "failed_segment_index": 0,
                "executed_transition_ids": (),
                "replay_step_ids": (),
            }
        )
        .model_dump_json(indent=2)
    )

    report = build_replay_usage_report(tmp_path)

    assert report.evidence_run_ids == ("run_1", "run_2")
    assert len(report.flows) == 1
    usage = report.flows[0]
    assert usage.attempt_count == 2
    assert usage.completed_count == 1
    assert usage.diverged_count == 1
    assert usage.rejoined_count == 1
    assert usage.attempted_run_ids == ("run_1", "run_2")
    assert usage.completed_run_ids == ("run_1",)
    assert usage.diverged_run_ids == ("run_2",)
    assert usage.rejoined_run_ids == ("run_2",)
    assert usage.repeated_failed_transition_id is None


def test_usage_report_identifies_one_repeated_failure_on_a_never_completed_path(
    tmp_path: Path,
) -> None:
    for run_id in ("run_1", "run_2"):
        evidence = ReplayRunEvidence(
            run_id=run_id,
            path_id="checkpoint_1",
            path_kind=ReplayPathKind.DISCOVERY_CHECKPOINT,
            terminal=RouteReplayTerminal.DIVERGED,
            completed_segment_count=1,
            failed_segment_index=1,
            replay_transition_ids=("transition_1", "transition_2", "transition_3"),
        )
        path = tmp_path / run_id / REPLAY_EVIDENCE_PATH
        path.parent.mkdir(parents=True)
        path.write_text(evidence.model_dump_json())

    report = build_replay_usage_report(tmp_path)

    assert repeated_failed_transition(report, "checkpoint_1") == "transition_2"


def test_historical_snapshot_accepts_current_run_without_an_archive_rescan(
    tmp_path: Path,
) -> None:
    historical = ReplayRunEvidence(
        run_id="run_1",
        path_id="checkpoint_1",
        path_kind=ReplayPathKind.DISCOVERY_CHECKPOINT,
        terminal=RouteReplayTerminal.DIVERGED,
        completed_segment_count=0,
        failed_segment_index=0,
        replay_transition_ids=("transition_1",),
    )
    path = tmp_path / historical.run_id / REPLAY_EVIDENCE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(historical.model_dump_json())
    snapshot = load_replay_evidence_snapshot(tmp_path)
    current = historical.model_copy(update={"run_id": "run_2"})

    report = build_replay_usage_report_from_snapshot(snapshot, current)

    assert report.evidence_run_ids == ("run_1", "run_2")
    assert repeated_failed_transition(report, "checkpoint_1") == "transition_1"
