"""Discovery terminal isolation for post-run procedure compilation."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent.discovery import DiscoveryResult
from yewo.agent.discovery.outcome import (
    persist_discovery_outcome_and_compile,
)
from yewo.agent.memory.records import MemoryCompatibility
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.memory.store import AppMemoryStore
from yewo.models import ModelClient
from yewo.agent.replay.procedure.records import ProcedureReplayTerminal
from yewo.runs import RunExecutionContext, RunTerminalCategory, RunWorkflowOutcome
from yewo.extensions import ExtensionRegistry


def _context(tmp_path: Path) -> RunExecutionContext:
    return cast(
        RunExecutionContext,
        SimpleNamespace(
            extensions=ExtensionRegistry(),
            workspace=SimpleNamespace(
                root=tmp_path,
                start=SimpleNamespace(run_id="run_1"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "category",
    (
        RunTerminalCategory.INCOMPLETE,
        RunTerminalCategory.BUDGET_EXHAUSTED,
        RunTerminalCategory.APPLICATION_FAILURE,
    ),
)
def test_non_completed_discovery_never_compiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: RunTerminalCategory,
) -> None:
    expected = RunWorkflowOutcome(category, "The source run did not complete.")
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.persist_discovery_outcome",
        lambda *args: expected,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.compile_completed_discovery_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("compilation called")),
    )

    outcome = persist_discovery_outcome_and_compile(
        _context(tmp_path),
        cast(DiscoveryResult, SimpleNamespace(procedure_replay=None)),
        True,
        cast(ModelClient, object()),
        "reconciler_primary",
        ScreenComparisonPolicy(),
    )

    assert outcome is expected


def test_compilation_failure_does_not_change_completed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RunWorkflowOutcome(RunTerminalCategory.COMPLETED, "The source run completed.")
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.persist_discovery_outcome",
        lambda *args: expected,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.compile_completed_discovery_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("compilation failure")),
    )

    outcome = persist_discovery_outcome_and_compile(
        _context(tmp_path),
        cast(DiscoveryResult, SimpleNamespace(procedure_replay=None)),
        True,
        cast(ModelClient, object()),
        "reconciler_primary",
        ScreenComparisonPolicy(),
    )

    assert outcome is expected


def test_completed_procedure_replay_does_not_recompile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RunWorkflowOutcome(RunTerminalCategory.COMPLETED, "Replay completed.")
    result = cast(
        DiscoveryResult,
        SimpleNamespace(
            procedure_replay=SimpleNamespace(terminal=ProcedureReplayTerminal.COMPLETED)
        ),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.persist_discovery_outcome",
        lambda *args: expected,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.compile_completed_discovery_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("compilation called")),
    )

    outcome = persist_discovery_outcome_and_compile(
        _context(tmp_path),
        result,
        True,
        cast(ModelClient, object()),
        "reconciler_primary",
        ScreenComparisonPolicy(),
    )

    assert outcome is expected


def test_writeback_failure_does_not_change_completed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RunWorkflowOutcome(RunTerminalCategory.COMPLETED, "The source run completed.")
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.persist_discovery_outcome",
        lambda *args: expected,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.compile_completed_discovery_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("compilation failure")),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.outcome.write_back_discovery_knowledge",
        lambda *args: (_ for _ in ()).throw(RuntimeError("write-back failure")),
    )

    outcome = persist_discovery_outcome_and_compile(
        _context(tmp_path),
        cast(DiscoveryResult, SimpleNamespace(procedure_replay=None)),
        True,
        cast(ModelClient, object()),
        "reconciler_primary",
        ScreenComparisonPolicy(),
        compatibility=cast(MemoryCompatibility, object()),
        memory_store=cast(AppMemoryStore, object()),
    )

    assert outcome is expected
