"""Sequential target selection, run isolation, and execution summaries."""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from yewo.errors import UsageError
from yewo.testing.lifecycle import LifecyclePhase, LifecycleStatus
from yewo.testing.models import JourneyReport, ReportStatus
from yewo.testing.report.page import theme
from yewo.testing.report.page.statuses import TONES
from yewo.testing.report.reports import safe_artifact_stem, write_atomic
from yewo.testing.suite import Target, Workflow


@dataclass(frozen=True)
class WorkflowTargets:
    """One workflow and the selected targets that will receive it."""

    workflow: Workflow
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class TargetRunResult:
    """The small cross-target record retained by an execution summary."""

    workflow: str
    target_id: str
    journey_id: str
    journey_name: str
    run_id: str
    status: str
    platform: str
    device: str
    duration_seconds: float = 0.0
    app_sha256: Optional[str] = None
    report_path: Optional[Path] = None
    error: Optional[str] = None

    @classmethod
    def from_report(
        cls,
        workflow: str,
        target: Target,
        run_id: str,
        report: JourneyReport,
        report_path: Optional[Path],
    ) -> "TargetRunResult":
        environment = report.environment
        app_sha256 = environment.app.artifact_sha256 if environment else None
        device_model = environment.device.model if environment else None
        return cls(
            workflow=workflow,
            target_id=target.id,
            journey_id=report.journey_id,
            journey_name=report.journey_name,
            run_id=run_id,
            status=str(report.status),
            platform=str(target.settings["platform"]),
            device=device_model or "Not recorded",
            duration_seconds=report.total_duration,
            app_sha256=app_sha256,
            report_path=report_path,
        )


def uses_target_matrix(workflows: Sequence[Workflow]) -> bool:
    """Whether the selected suite declared at least one explicit target."""
    return any(workflow.targets for workflow in workflows)


def select_workflow_targets(
    workflows: Sequence[Workflow], wanted: Optional[str]
) -> list[WorkflowTargets]:
    """Select targets globally while preserving workflow and declaration order."""
    requested = tuple(name.strip() for name in (wanted or "").split(",") if name.strip())
    unnamed = [workflow.name or "default" for workflow in workflows if not workflow.targets]
    if unnamed:
        raise UsageError(
            "Every selected workflow must declare targets in a matrix run. Missing: "
            f"{', '.join(unnamed)}"
        )
    known = {
        target.id
        for workflow in workflows
        for target in workflow.targets
    }
    missing = [name for name in requested if name not in known]
    if missing:
        raise UsageError(
            f"No target named {', '.join(missing)}. "
            f"This suite has: {', '.join(sorted(known)) or '(none)'}"
        )

    selected = []
    for workflow in workflows:
        targets = tuple(
            target for target in workflow.targets if not requested or target.id in requested
        )
        if targets:
            selected.append(WorkflowTargets(workflow, targets))
    return selected


def validate_matrix_arguments(args) -> None:
    """Refuse flags that would silently replace every declared target."""
    if not args.locked:
        raise UsageError(
            "A targets matrix requires --locked so every target runs the same suite. "
            "Run 'yewo lock' first."
        )
    conflicting = [
        flag
        for flag, value in (
            ("--device", args.device),
            ("--platform", args.platform),
            ("--app-path", args.app_path),
        )
        if value is not None
    ]
    if conflicting:
        raise UsageError(
            f"{', '.join(conflicting)} cannot override a targets matrix. "
            "Select a declared target with --target."
        )


def isolated_run_dir(
    execution_root: Path,
    workflow: str,
    target_id: str,
    journey_id: str,
    run_id: str,
) -> Path:
    """The artifact root owned by one journey and target attempt."""
    return (
        execution_root
        / safe_artifact_stem(workflow or "default")
        / safe_artifact_stem(target_id)
        / safe_artifact_stem(journey_id)
        / run_id
    )


def restoration_failed(report: JourneyReport) -> bool:
    """Whether this run left its target unsafe for another matrix item."""
    return any(
        operation.phase is LifecyclePhase.RESTORATION
        and operation.status is LifecycleStatus.FAILED
        for scenario in report.scenarios
        for operation in scenario.lifecycle_operations
    )


def _report_href(root: Path, result: TargetRunResult) -> Optional[str]:
    if result.report_path is None:
        return None
    try:
        return result.report_path.relative_to(root).as_posix()
    except ValueError:
        return result.report_path.resolve().as_uri()


def _payload(
    execution_id: str,
    expected_runs: int,
    results: Sequence[TargetRunResult],
    root: Path,
) -> dict:
    complete = len(results) == expected_runs
    passed = complete and expected_runs > 0 and all(
        result.status == ReportStatus.PASSED for result in results
    )
    interrupted = any(result.status == ReportStatus.ABORTED for result in results)
    return {
        "schema": 1,
        "execution_id": execution_id,
        "status": (
            "PASSED"
            if passed
            else "ABORTED"
            if interrupted
            else "FAILED"
            if complete
            else "RUNNING"
        ),
        "expected_runs": expected_runs,
        "completed_runs": len(results),
        "runs": [
            {
                "workflow": result.workflow,
                "target_id": result.target_id,
                "journey_id": result.journey_id,
                "journey_name": result.journey_name,
                "run_id": result.run_id,
                "status": result.status,
                "platform": result.platform,
                "device": result.device,
                "duration_seconds": result.duration_seconds,
                "app_sha256": result.app_sha256,
                "report": _report_href(root, result),
                "error": result.error,
            }
            for result in results
        ],
    }


def _status_tone(status: str) -> str:
    try:
        return TONES[ReportStatus(status)]
    except ValueError:
        return "warn"


def _status_badge(status: str) -> str:
    tone = _status_tone(status)
    return f'<span class="badge b-{tone}">{html.escape(status)}</span>'


def _run_cards(payload: dict) -> str:
    cards = []
    for run in payload["runs"]:
        report = run["report"]
        report_link = (
            f'<a class="run-link" href="{html.escape(report, quote=True)}">Open run report</a>'
            if report
            else '<span class="run-unavailable">Run report unavailable</span>'
        )
        app = (run["app_sha256"] or "")[:12] or "Not recorded"
        target = html.escape(run["target_id"])
        error = (
            f'<p class="failure-reason">{html.escape(run["error"])}</p>'
            if run["error"]
            else ""
        )
        cards.append(
            '<article class="run-card">'
            '<div class="head"><div>'
            f'<h3>{html.escape(run["journey_name"])}</h3>'
            f'<div class="run-target mono">{target}</div></div>'
            f'{_status_badge(run["status"])}</div>'
            '<dl class="run-facts">'
            f'<div><dt>Workflow</dt><dd>{html.escape(run["workflow"] or "default")}</dd></div>'
            f'<div><dt>Platform</dt><dd>{html.escape(run["platform"])}</dd></div>'
            f'<div><dt>Device</dt><dd>{html.escape(run["device"])}</dd></div>'
            f'<div><dt>App hash</dt><dd class="mono">{html.escape(app)}</dd></div>'
            f'<div><dt>Run ID</dt><dd class="mono">{html.escape(run["run_id"])}</dd></div>'
            f'<div><dt>Duration</dt><dd>{run["duration_seconds"]:.1f} seconds</dd></div>'
            f'</dl>{error}<div class="run-actions">{report_link}</div></article>'
        )
    return "\n".join(cards) or '<div class="empty-runs">No target runs have completed yet.</div>'


def _verdict_copy(payload: dict) -> str:
    return {
        "PASSED": "All target runs passed",
        "FAILED": "One or more target runs did not pass",
        "ABORTED": "Target execution was interrupted",
        "RUNNING": "Target execution is in progress",
    }.get(payload["status"], "Target execution status")


def _html_report(payload: dict) -> str:
    status = html.escape(payload["status"])
    execution_id = html.escape(payload["execution_id"])
    tone = _status_tone(payload["status"])
    completed = payload["completed_runs"]
    expected = payload["expected_runs"]
    cards = _run_cards(payload)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            '<head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Yewo execution report · {execution_id}</title>",
            f"<style>{theme.stylesheet()}\n{theme.execution_stylesheet()}</style></head>",
            '<body><header class="top"><div class="bar"><div class="brand">',
            f'<span class="logo">{theme.MARKS["logo"]}<span class="name">yewo</span></span>',
            '<span class="slash" aria-hidden="true">/</span>',
            '<span class="report-label">Execution report</span></div>',
            f'<div class="stamp"><span class="mono">{execution_id}</span>{theme.MARKS["toggle"]}</div>',
            '</div></header><main>',
            '<div class="title"><span class="muted">Target matrix</span>',
            '<h1>Sequential target execution</h1></div>',
            f'<section class="verdict t-{tone}"><div class="verdict-body">',
            f'<div class="line">{_status_badge(payload["status"])}',
            f'<h2>{html.escape(_verdict_copy(payload))}</h2></div>',
            f'<p>{completed} of {expected} target runs completed.</p></div></section>',
            '<section class="tiles" aria-label="Execution summary">',
            f'<div class="tile"><span class="label">Status</span><span class="value">{status}</span></div>',
            f'<div class="tile"><span class="label">Completed runs</span><span class="value">{completed} / {expected}</span></div>',
            f'<div class="tile"><span class="label">Execution ID</span><span class="value mono">{execution_id}</span></div>',
            '</section><section class="execution-runs"><div class="head">',
            '<h2>Target runs</h2><span class="muted">Each target keeps its own evidence report.</span>',
            f'</div><div class="run-grid">{cards}</div></section>',
            '<footer><span>This execution index works offline with its saved run reports.</span></footer>',
            f'</main><script>{theme.script()}</script></body></html>',
        ]
    )


def _junit_report(results: Iterable[TargetRunResult]) -> bytes:
    listed = list(results)
    error_statuses = {ReportStatus.ERROR, ReportStatus.ABORTED, "ERROR", "ABORTED"}
    errors = sum(result.status in error_statuses for result in listed)
    failures = sum(
        result.status != ReportStatus.PASSED and result.status not in error_statuses
        for result in listed
    )
    suite = ET.Element(
        "testsuite",
        name="yewo.target_matrix",
        tests=str(len(listed)),
        failures=str(failures),
        errors=str(errors),
        skipped="0",
        time=str(sum(result.duration_seconds for result in listed)),
    )
    for result in listed:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"{safe_artifact_stem(result.workflow or 'default')}.{result.target_id}",
            name=result.journey_id,
            time=str(result.duration_seconds),
        )
        if result.status != ReportStatus.PASSED:
            element = "error" if result.status in error_statuses else "failure"
            problem = ET.SubElement(case, element, message=result.status)
            problem.text = result.error or f"Target run ended {result.status}."
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


def write_execution_summary(
    root: Path,
    execution_id: str,
    expected_runs: int,
    results: Sequence[TargetRunResult],
) -> tuple[Path, Path, Path]:
    """Atomically checkpoint the execution index after every target run."""
    root.mkdir(parents=True, exist_ok=True)
    payload = _payload(execution_id, expected_runs, results, root)
    json_path = root / "execution.json"
    html_path = root / "index.html"
    junit_path = root / "junit.xml"
    write_atomic(json_path, json.dumps(payload, indent=2) + "\n")
    write_atomic(html_path, _html_report(payload))
    write_atomic(junit_path, _junit_report(results))
    return html_path, json_path, junit_path
