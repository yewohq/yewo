"""The summary tiles under the verdict: what the run cost, and what produced it.

Separate from the page's account of what happened, because these answer a
different question. A reader comes here when two runs of one journey disagree
and the journey did not change.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from yewo.testing.models import RunEnvironment

from .source import RunSource


@dataclass(frozen=True)
class SummaryTile:
    """One fact about the run, in the band under the verdict."""

    label: str
    value: str
    note: str = ""


def seconds(value: float) -> str:
    """A duration a reader compares at a glance, in the unit that suits it."""
    if value >= 60:
        minutes, rest = divmod(int(round(value)), 60)
        return f"{minutes} min {rest} s"
    if value >= 10:
        return f"{value:.0f} s"
    return f"{value:.1f} s"


def _build_cost_tile(usage: Optional[dict]) -> SummaryTile:
    """What the run spent, and how much of it carried a price.

    A call whose price nothing knew is counted separately rather than as
    zero. The total would otherwise read as complete.
    """
    if not usage:
        return SummaryTile("Estimated Cost", "Not recorded", "no usage telemetry")
    estimated_cost = usage.get("total_estimated_cost_usd") or 0.0
    calls = usage.get("total_calls") or 0
    unknown = usage.get("cost_unknown_calls") or 0
    note = f"{calls} model calls"
    if unknown:
        note = f"{calls} model calls · {unknown} without a price"
    return SummaryTile("Estimated Cost", f"${estimated_cost:.2f}", note)


def _format_scenario_count_label(scenarios: int) -> str:
    """How many scenarios the step count covers, said once correctly."""
    return "across 1 scenario" if scenarios == 1 else f"across {scenarios} scenarios"


def _build_device_tile(environment: Optional[RunEnvironment]) -> SummaryTile:
    """Which target, and which one of several like it.

    The serial is already masked to its last few characters where it is
    read. It is here because two emulators of the same model are the same
    tile without it.
    """
    if environment is None:
        return SummaryTile("Device", "Not recorded")
    device = environment.device
    # A bare "16" beside a model name could be the Android release, the API
    # level or an iOS version.
    release = f"{_format_platform_name(environment.platform)} {device.os_version}".strip()
    device_label = " · ".join(part for part in (device.model, release) if part)
    note = " · ".join(
        part
        for part in ("emulator" if device.emulator else "", device.serial or "")
        if part
    )
    return SummaryTile("Device", device_label or environment.platform, note)


def _format_platform_name(platform: str) -> str:
    """The platform as a reader names it."""
    return {"android": "Android", "ios": "iOS"}.get((platform or "").lower(), "")


def _build_app_tile(environment: Optional[RunEnvironment]) -> SummaryTile:
    """Which build was tested, and whether this run is the one that installed it."""
    if environment is None:
        return SummaryTile("Build", "Not recorded")
    app = environment.app
    app_label = app.version or app.package or "Not recorded"
    if app.version and app.build:
        app_label = f"{app.version} ({app.build})"
    # The package names which app, and the artifact hash names which copy of
    # it. A version string is a claim the build makes about itself.
    # A dirty checkout is named, because a run of edited files is not
    # evidence about the commit beside it.
    commit = (environment.git.sha or "")[:7]
    if commit and environment.git.dirty:
        # Named, so two dirty runs of one commit are told apart. Without it
        # a reader sees the same line for two different working trees.
        edited = (environment.git.diff_sha256 or "")[:8]
        commit = f"{commit} + uncommitted {edited}".strip()
    parts = [
        app.package or "",
        (app.artifact_sha256 or "")[:12],
        commit,
    ]
    if not app.installed:
        parts.append("already installed")
    return SummaryTile("Build", app_label, " · ".join(part for part in parts if part))


def _truncate_revision(revision_name: str) -> str:
    """A pinned revision at the length a reader compares, like a git sha."""
    repository, _, revision = revision_name.partition("@")
    return f"{repository}@{revision[:8]}" if revision else revision_name


def _build_grounding_tile(environment: Optional[RunEnvironment]) -> SummaryTile:
    """How the planner named a target, which changes how a run walks."""
    if environment is None:
        return SummaryTile("Targeting", "Not recorded")
    return SummaryTile("Targeting", environment.grounding, _truncate_revision(environment.detector or ""))


def _build_branch_tile(environment: Optional[RunEnvironment]) -> SummaryTile:
    """Where the commit came from, and the job that ran it."""
    if environment is None or not environment.git.branch:
        return SummaryTile("Branch", "Not recorded")
    return SummaryTile("Branch", environment.git.branch, environment.ci.provider or "local")


def tiles_of(report: RunSource, usage: Optional[dict]) -> Tuple[SummaryTile, ...]:
    """Every tile, in the order the band reads.

    One band rather than a band and a footer line. What a run was given
    and what it came to answer the same question: why this run and the
    last one differ. Splitting them put half the answer under the page.
    """
    environment = report.environment
    steps = sum(scenario.steps_executed for scenario in report.scenarios)
    return (
        SummaryTile(
            "Scenarios",
            f"{report.passed_count} of {len(report.scenarios)} passed",
            f"{report.skipped_count} skipped" if report.skipped_count else "",
        ),
        SummaryTile("Steps", str(steps), _format_scenario_count_label(len(report.scenarios))),
        SummaryTile("Duration", seconds(report.total_duration) or "0 s"),
        _build_cost_tile(usage),
        SummaryTile(
            "Model",
            environment.model if environment else "Not recorded",
            f"fast: {environment.fast_model}"
            if environment and environment.fast_model
            else "",
        ),
        _build_device_tile(environment),
        _build_grounding_tile(environment),
        _build_branch_tile(environment),
        _build_app_tile(environment),
    )
