"""Canonical replay evidence and derived cross-run usage reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.agent.route_replay import ReplayPathKind, RouteReplayResult, RouteReplayTerminal
from yewo.artifacts import ARTIFACTS_DIRECTORY, AtomicArtifactWriter, RecordIdentifier

if TYPE_CHECKING:
    from yewo.agent.discovery.records import DiscoveryResult


REPLAY_EVIDENCE_PATH = ARTIFACTS_DIRECTORY / "agent" / "replay" / "evidence.json"
REPLAY_ATTEMPT_PATH = ARTIFACTS_DIRECTORY / "agent" / "replay" / "attempt.json"
REPLAY_REPORT_PATH = ARTIFACTS_DIRECTORY / "agent" / "replay" / "report.json"


class ReplayRunEvidence(BaseModel):
    """One run's references to canonical stored-flow execution evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: RecordIdentifier
    path_id: RecordIdentifier
    path_kind: ReplayPathKind
    terminal: RouteReplayTerminal
    completed_segment_count: int = Field(ge=0)
    failed_segment_index: int | None = Field(default=None, ge=0)
    replay_transition_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    executed_transition_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    replay_step_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    rejoined_transition_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    rejoined_step_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def _validate_evidence_references(self) -> Self:
        if not set(self.executed_transition_ids).issubset(self.replay_transition_ids):
            raise ValueError("Executed replay transitions must belong to the selected flow.")
        if bool(self.rejoined_transition_ids) != bool(self.rejoined_step_ids):
            raise ValueError("A downstream rejoin requires transitions and step evidence.")
        return self

    @property
    def full_replay_succeeded(self) -> bool:
        """Return whether every selected replay segment passed its live guards."""
        return (
            bool(self.replay_transition_ids)
            and self.terminal is RouteReplayTerminal.COMPLETED
            and self.completed_segment_count == len(self.replay_transition_ids)
            and self.executed_transition_ids == self.replay_transition_ids
        )


@dataclass(frozen=True)
class ReplayEvidenceSnapshot:
    """Resolved historical replay evidence loaded once for one discovery run."""

    evidence: tuple[ReplayRunEvidence, ...]
    unreadable_evidence_paths: tuple[str, ...] = ()
    duplicate_run_ids: tuple[RecordIdentifier, ...] = ()


class ReplayFlowUsage(BaseModel):
    """Derived unique run identifiers grouped by one retained flow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_id: RecordIdentifier
    path_kind: ReplayPathKind
    attempt_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    diverged_count: int = Field(ge=0)
    rejoined_count: int = Field(ge=0)
    repeated_failed_transition_id: RecordIdentifier | None = None
    attempted_run_ids: tuple[RecordIdentifier, ...]
    completed_run_ids: tuple[RecordIdentifier, ...]
    diverged_run_ids: tuple[RecordIdentifier, ...]
    rejoined_run_ids: tuple[RecordIdentifier, ...]

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        counts = (
            (self.attempt_count, self.attempted_run_ids),
            (self.completed_count, self.completed_run_ids),
            (self.diverged_count, self.diverged_run_ids),
            (self.rejoined_count, self.rejoined_run_ids),
        )
        if any(count != len(run_ids) for count, run_ids in counts):
            raise ValueError("Replay usage counts must match their unique run identifiers.")
        return self


class ReplayUsageReport(BaseModel):
    """A rebuildable summary whose claims resolve to per-run evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evidence_run_ids: tuple[RecordIdentifier, ...]
    flows: tuple[ReplayFlowUsage, ...]
    unreadable_evidence_paths: tuple[str, ...] = ()
    duplicate_run_ids: tuple[RecordIdentifier, ...] = ()


def build_replay_run_evidence(
    run_id: RecordIdentifier,
    result: DiscoveryResult,
) -> ReplayRunEvidence | None:
    """Build replay references without duplicating action or observation evidence."""
    replay = result.replay
    if replay is None:
        return None
    evidence = _build_replay_evidence(run_id, replay)
    replay_step_ids = tuple(step.step_id for step in replay.agent_result.steps)
    replay_step_id_set = frozenset(replay_step_ids)
    rejoined_steps = tuple(
        step
        for step in result.agent_result.steps
        if step.step_id not in replay_step_id_set
        and step.decision.reused_flow_id == replay.path_id
        and step.decision.reused_transition_ids
    )
    return evidence.model_copy(
        update={
            "rejoined_transition_ids": tuple(
                dict.fromkeys(
                    transition_id
                    for step in rejoined_steps
                    for transition_id in step.decision.reused_transition_ids
                )
            ),
            "rejoined_step_ids": tuple(step.step_id for step in rejoined_steps),
        }
    )


def build_initial_replay_evidence(
    run_id: RecordIdentifier,
    replay: RouteReplayResult,
) -> ReplayRunEvidence:
    """Build evidence before repair so interruption cannot erase the attempt."""
    return _build_replay_evidence(run_id, replay)


def build_replay_usage_report(search_root: Path) -> ReplayUsageReport:
    """Derive retained-flow usage from immutable per-run evidence artifacts."""
    return build_replay_usage_report_from_snapshot(load_replay_evidence_snapshot(search_root))


def load_replay_evidence_snapshot(search_root: Path) -> ReplayEvidenceSnapshot:
    """Load and resolve immutable replay evidence once for one discovery run."""
    evidence_by_run_id: dict[RecordIdentifier, ReplayRunEvidence] = {}
    priority_by_run_id: dict[RecordIdentifier, int] = {}
    unreadable: list[str] = []
    duplicate_run_ids: set[RecordIdentifier] = set()
    evidence_paths = (
        (0, sorted(search_root.rglob(REPLAY_ATTEMPT_PATH.as_posix()))),
        (1, sorted(search_root.rglob(REPLAY_EVIDENCE_PATH.as_posix()))),
    )
    for priority, paths in evidence_paths:
        for path in paths:
            try:
                evidence = ReplayRunEvidence.model_validate_json(path.read_text())
            except (OSError, UnicodeError, ValidationError):
                unreadable.append(_evidence_path_label(search_root, path))
                continue
            previous_priority = priority_by_run_id.get(evidence.run_id)
            if previous_priority is None or priority > previous_priority:
                priority_by_run_id[evidence.run_id] = priority
                evidence_by_run_id[evidence.run_id] = evidence
                duplicate_run_ids.discard(evidence.run_id)
            elif priority == previous_priority:
                duplicate_run_ids.add(evidence.run_id)
                evidence_by_run_id.pop(evidence.run_id, None)
    return ReplayEvidenceSnapshot(
        evidence=tuple(evidence_by_run_id[run_id] for run_id in sorted(evidence_by_run_id)),
        unreadable_evidence_paths=tuple(sorted(set(unreadable))),
        duplicate_run_ids=tuple(sorted(duplicate_run_ids)),
    )


def build_replay_usage_report_from_snapshot(
    snapshot: ReplayEvidenceSnapshot,
    current_run_evidence: ReplayRunEvidence | None = None,
) -> ReplayUsageReport:
    """Build usage from one historical snapshot plus optional current-run evidence."""
    evidence_by_run_id = {item.run_id: item for item in snapshot.evidence}
    duplicate_run_ids = set(snapshot.duplicate_run_ids)
    if current_run_evidence is not None:
        evidence_by_run_id[current_run_evidence.run_id] = current_run_evidence
        duplicate_run_ids.discard(current_run_evidence.run_id)
    flows: list[ReplayFlowUsage] = []
    flow_keys = sorted({(item.path_id, item.path_kind) for item in evidence_by_run_id.values()})
    for path_id, path_kind in flow_keys:
        matches = tuple(
            item
            for item in evidence_by_run_id.values()
            if item.path_id == path_id and item.path_kind is path_kind
        )
        attempted_run_ids = _sorted_run_ids(matches)
        completed_run_ids = _sorted_run_ids(item for item in matches if item.full_replay_succeeded)
        diverged_run_ids = _sorted_run_ids(
            item for item in matches if item.terminal is RouteReplayTerminal.DIVERGED
        )
        rejoined_run_ids = _sorted_run_ids(item for item in matches if item.rejoined_step_ids)
        failed_transition_counts = Counter(
            item.replay_transition_ids[item.failed_segment_index]
            for item in matches
            if item.terminal is RouteReplayTerminal.DIVERGED
            and item.failed_segment_index is not None
            and item.failed_segment_index < len(item.replay_transition_ids)
        )
        repeated_failed_transition_id = next(
            (
                transition_id
                for transition_id, count in failed_transition_counts.most_common()
                if count >= 2 and not completed_run_ids
            ),
            None,
        )
        flows.append(
            ReplayFlowUsage(
                path_id=path_id,
                path_kind=path_kind,
                attempt_count=len(attempted_run_ids),
                completed_count=len(completed_run_ids),
                diverged_count=len(diverged_run_ids),
                rejoined_count=len(rejoined_run_ids),
                repeated_failed_transition_id=repeated_failed_transition_id,
                attempted_run_ids=attempted_run_ids,
                completed_run_ids=completed_run_ids,
                diverged_run_ids=diverged_run_ids,
                rejoined_run_ids=rejoined_run_ids,
            )
        )
    return ReplayUsageReport(
        evidence_run_ids=tuple(sorted(evidence_by_run_id)),
        flows=tuple(flows),
        unreadable_evidence_paths=snapshot.unreadable_evidence_paths,
        duplicate_run_ids=tuple(sorted(duplicate_run_ids)),
    )


def repeated_failed_transition(
    report: ReplayUsageReport,
    path_id: RecordIdentifier,
) -> RecordIdentifier | None:
    """Return a repeatedly divergent transition for one never-completed path."""
    return next(
        (
            flow.repeated_failed_transition_id
            for flow in report.flows
            if flow.path_id == path_id
        ),
        None,
    )


def load_replay_failure_snapshot(
    search_root: Path,
    path_id: RecordIdentifier,
) -> tuple[ReplayEvidenceSnapshot, RecordIdentifier | None]:
    """Load historical evidence once and select one repeated path failure."""
    snapshot = load_replay_evidence_snapshot(search_root)
    report = build_replay_usage_report_from_snapshot(snapshot)
    return snapshot, repeated_failed_transition(report, path_id)


def persist_replay_evidence(
    run_root: Path,
    run_id: RecordIdentifier,
    result: DiscoveryResult,
    historical_snapshot: ReplayEvidenceSnapshot | None = None,
) -> None:
    """Persist this run's evidence and a rebuildable output-root report."""
    writer = AtomicArtifactWriter(run_root)
    evidence = build_replay_run_evidence(run_id, result)
    if evidence is None:
        return
    writer.write(REPLAY_EVIDENCE_PATH, _json_bytes(evidence))
    report = (
        build_replay_usage_report_from_snapshot(historical_snapshot, evidence)
        if historical_snapshot is not None
        else build_replay_usage_report(run_root.parent)
    )
    writer.write(REPLAY_REPORT_PATH, _json_bytes(report))


def persist_initial_replay_evidence(
    run_root: Path,
    run_id: RecordIdentifier,
    replay: RouteReplayResult,
) -> None:
    """Persist the replay attempt before optional model repair begins."""
    writer = AtomicArtifactWriter(run_root)
    writer.write(REPLAY_ATTEMPT_PATH, _json_bytes(_build_replay_evidence(run_id, replay)))


def _build_replay_evidence(
    run_id: RecordIdentifier,
    replay: RouteReplayResult,
) -> ReplayRunEvidence:
    replay_steps = tuple(
        step for step in replay.agent_result.steps if step.decision.reused_transition_ids
    )
    return ReplayRunEvidence(
        run_id=run_id,
        path_id=replay.path_id,
        path_kind=replay.path_kind,
        terminal=replay.terminal,
        completed_segment_count=replay.completed_segment_count,
        failed_segment_index=replay.failed_segment_index,
        replay_transition_ids=replay.replay_transition_ids,
        executed_transition_ids=tuple(
            transition_id
            for step in replay_steps
            for transition_id in step.decision.reused_transition_ids
        ),
        replay_step_ids=tuple(step.step_id for step in replay_steps),
    )


def _sorted_run_ids(
    evidence: Iterable[ReplayRunEvidence],
) -> tuple[RecordIdentifier, ...]:
    return tuple(sorted(item.run_id for item in evidence))


def _json_bytes(model: BaseModel) -> bytes:
    return model.model_dump_json(indent=2).encode("utf-8") + b"\n"


def _evidence_path_label(search_root: Path, path: Path) -> str:
    try:
        return path.relative_to(search_root).as_posix()
    except ValueError:
        return path.as_posix()
