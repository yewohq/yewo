"""Render one canonical run record as Markdown."""

from pathlib import PurePosixPath

from yewo.reporting.records import CanonicalRunRecord


def render_markdown(record: CanonicalRunRecord) -> str:
    """Return a concise report without recomputing any verdict."""
    lines = [
        "# Yewo test report",
        "",
        f"- **Run:** `{record.run_id}`",
        f"- **QA verdict:** **{record.qa_verdict.value.upper()}**",
        f"- **Completeness:** {record.completeness.value}",
        f"- **Terminal:** {record.terminal_category.value} at {record.terminal_stage.value}",
        f"- **Reason:** {_plain(record.terminal_reason)}",
        f"- **Duration:** {(record.completed_at - record.started_at).total_seconds():.2f}s",
        "",
        "## Scenario results",
        "",
    ]
    if not record.scenarios:
        lines.append("No executable scenarios produced an audited result.")
    for scenario in record.scenarios:
        lines.extend(
            [
                f"### {scenario.audit.verdict.value.upper()} — {_plain(scenario.name)}",
                "",
                f"- **Coverage:** `{scenario.coverage_cell_id}`",
                f"- **Execution:** `{scenario.execution.classification.value}`",
                f"- **Objective:** {_plain(scenario.objective)}",
                f"- **Audit:** {_plain(scenario.audit.summary)}",
                "",
                "| Expected outcome | Result | Reason |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_cell(item.description)} | {item.status.value} | {_cell(item.reason)} |"
            for item in scenario.audit.obligations
        )
        if scenario.interactions:
            lines.extend(["", "#### Interaction evidence", ""])
            for interaction in scenario.interactions:
                evidence = next(
                    item
                    for item in record.evidence
                    if item.artifact_id == interaction.screenshot_evidence_id
                )
                geometry = f"tap ({interaction.point.x}, {interaction.point.y})"
                if interaction.region is not None:
                    region = interaction.region
                    geometry += (
                        f" inside [{region.left}, {region.top}, {region.right}, {region.bottom}]"
                    )
                target = f" — {_plain(interaction.target)}" if interaction.target else ""
                href = PurePosixPath("..") / PurePosixPath(evidence.path.as_posix())
                lines.append(
                    f"- `{interaction.action}` {geometry}{target} "
                    f"— `{interaction.outcome_disposition or interaction.dispatch_disposition}` "
                    f"([source screenshot]({href.as_posix()}))"
                )
        lines.append("")
    usage = record.usage
    lines.extend(
        [
            "## Model usage",
            "",
            f"- **Calls:** {usage.model_call_count}",
            f"- **Calls with unknown usage:** {usage.calls_with_unknown_usage}",
            f"- **Known prompt tokens:** {usage.known_prompt_tokens}",
            f"- **Known completion tokens:** {usage.known_completion_tokens}",
            f"- **Known cached tokens:** {usage.known_cached_tokens}",
            f"- **Known cost:** ${usage.known_cost_usd}",
            "",
        ]
    )
    if record.model_calls:
        lines.extend(
            [
                "### Model calls",
                "",
                "| Role | Profile | Model | Result | Prompt | Completion | Cost |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {_cell(call.role)} | {_cell(call.profile_id)} | "
            f"{_cell(call.served_model or call.requested_model)} | {_cell(call.disposition)} | "
            f"{call.prompt_tokens if call.prompt_tokens is not None else 'unknown'} | "
            f"{call.completion_tokens if call.completion_tokens is not None else 'unknown'} | "
            f"{call.cost_usd if call.cost_usd is not None else 'unknown'} |"
            for call in record.model_calls
        )
        lines.append("")
    lines.extend(
        [
            "## Evidence",
            "",
        ]
    )
    for evidence in record.evidence:
        href = PurePosixPath("..") / PurePosixPath(evidence.path.as_posix())
        lines.append(f"- [{_plain(evidence.label)}]({href.as_posix()}) — `{evidence.kind.value}`")
    return "\n".join(lines).rstrip() + "\n"


def _plain(value: str | None) -> str:
    return (value or "").replace("\r", " ").replace("\n", " ").strip()


def _cell(value: str) -> str:
    return _plain(value).replace("|", "\\|")
