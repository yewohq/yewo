"""The run report as one HTML file, written where a reader will open it.

Standalone on purpose. A CI job uploads one file. Whoever opens it can be
offline, on a machine with no fonts installed, or behind a policy that blocks
other hosts. So nothing here is fetched. The stylesheet, the marks and the
small script that remembers a theme are all written into the page.

Screenshots are the exception. They stay beside the report as files. A run's
worth of inlined images makes a file too large to attach. The links are
relative, so the report travels with the directory around it.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Optional, Sequence

from yewo import __version__
from yewo.testing.models import JourneyReport
from yewo.utils.secrets import SecretScrubber

from ..reports import safe_artifact_stem, write_atomic

from . import theme
from .view import (
    CheckView,
    FindingView,
    ReportView,
    ScenarioView,
    StepView,
    page_of,
)

logger = logging.getLogger(__name__)

SECRET = re.compile(r"\{\{([A-Za-z0-9_.-]+)\}\}")
"""A masked value, shown as the name that declared it rather than as prose."""

OFFER_CLOUD = False
"""Whether the page offers Yewo Cloud at its foot.

Off, because the offer names a thing Yewo cannot do yet. `ReplayLedger`
holds a run's paths in memory and persists none of them. No step a run
recorded can be repeated by the next run. Durable replay is A9 in the plan.
Turn this on with it, and the offer becomes true of the report in front of
the reader.
"""

CLOUD = "https://yewo.dev/cloud"

CLOUD_LEAD = "Next time, the {count} steps that worked could replay without AI."
"""What the strip at the foot of the page offers, in the run's own numbers."""

CLOUD_BODY = (
    "Yewo Cloud remembers each path, compares a run with your last good build "
    "and posts the result on your pull request."
)
"""What Yewo Cloud adds to what this report already did."""


@dataclass(frozen=True)
class Markup:
    """One owner for every string that reaches the page.

    App content, model prose and journey text all arrive here. Each can carry
    markup, and each can carry a declared secret. Both are dealt with in one
    place. No call site can forget one of them.
    """

    scrubber: Optional[SecretScrubber] = None

    def text(self, value: Optional[str]) -> str:
        """Plain text, masked and escaped."""
        masked = self.scrubber.scrub(value) if self.scrubber else (value or "")
        return escape(masked, quote=True)

    def named(self, value: Optional[str]) -> str:
        """The same, with each masked value shown as the name that declared it."""
        return SECRET.sub(
            lambda found: f'<span class="named">{found.group(0)}</span>',
            self.text(value),
        )


def _mark(tone: str) -> str:
    return theme.ICONS.get(tone, theme.ICONS["muted"])


def _chips(step: StepView) -> str:
    return "".join(
        f'<span class="chip c-{chip.tone}">{escape(chip.label)}</span>'
        for chip in step.chips
    )


def _row(write: Markup, step: StepView, tag: str) -> str:
    """The line every step shows, open or closed."""
    took = f'<span class="took">{escape(step.duration)}</span>' if step.duration else ""
    caret = theme.MARKS["caret"] if tag == "summary" else ""
    return (
        f'<{tag} class="row">'
        f'<span class="n">{escape(step.number)}</span>{_mark(step.tone)}'
        f'<span class="what">{write.named(step.title)}</span>'
        f"{_chips(step)}{took}{caret}"
        f"</{tag}>"
    )


def _shots(write: Markup, step: StepView) -> str:
    images = [shot for shot in step.shots if shot.image]
    files = [shot for shot in step.shots if not shot.image]
    figures = "".join(
        f'<figure class="shot"><a href="{escape(shot.href, quote=True)}">'
        f'<img src="{escape(shot.href, quote=True)}" alt="{escape(shot.caption)}" loading="lazy">'
        f"</a><figcaption>{escape(shot.caption)}</figcaption></figure>"
        for shot in images
    )
    links = "".join(
        f'<a class="file" href="{escape(shot.href, quote=True)}">{escape(shot.caption)}</a>'
        for shot in files
    )
    if not figures and not links:
        return ""
    return f'<div class="shots">{figures}{links}</div>'


def _inside(write: Markup, step: StepView) -> str:
    notes = "".join(
        f"<div><dt>{escape(note.label)}</dt>"
        f"<dd>{write.named(note.body)}</dd></div>"
        for note in step.notes
    )
    written = f'<dl class="notes">{notes}</dl>' if notes else ""
    return f'<div class="inside">{_shots(write, step)}{written}</div>'


def _step(write: Markup, step: StepView) -> str:
    opened = " open" if step.opened else ""
    body = (
        f"<details{opened}>{_row(write, step, 'summary')}{_inside(write, step)}</details>"
        if step.expandable
        else _row(write, step, "div")
    )
    return f'<div class="step s-{step.tone}" id="{escape(step.anchor, quote=True)}">{body}</div>'


def _checks(write: Markup, checks: Sequence[CheckView]) -> str:
    if not checks:
        return ""
    rows = "".join(
        f'<div class="check"><span class="badge b-{check.tone}">{escape(check.status)}</span>'
        + (f'<span class="muted mono">{escape(check.named)}</span>' if check.named else "")
        + f'<span class="rule">{write.named(check.rule)}</span>'
        f'<span class="said">{write.named(check.reasoning)}'
        + (f' <span class="muted">· {write.text(check.screen)}</span>' if check.screen else "")
        + (
            f' <span class="muted">· {check.asked} observations evaluated</span>'
            if check.asked > 1
            else ""
        )
        + "</span></div>"
        for check in checks
    )
    return (
        '<section class="rows"><div class="head"><h2>Verification Results</h2>'
        '<span class="muted">Final result for each expected outcome and '
        "security rule</span></div>"
        f"{rows}</section>"
    )


def _walk(write: Markup, scenario: ScenarioView) -> str:
    """One scenario: its verdict, the steps it walked, the rules it answered."""
    why = (
        f'<p class="why t-{scenario.tone}">{write.named(scenario.failure_reason)}</p>'
        if scenario.failure_reason
        else ""
    )
    steps = "".join(_step(write, step) for step in scenario.steps)
    if not steps:
        steps = '<p class="muted">This scenario recorded no steps.</p>'
    summary = " · ".join(part for part in (scenario.status, scenario.summary) if part)
    # The cell and the test id are how a failure is matched to its lock
    # entry and to the same scenario in another run. Without them a reader
    # has a scenario name a model wrote, which two runs need not share.
    named = " · ".join(part for part in (scenario.cell, scenario.test_id) if part)
    identity = f'<span class="mono muted">{escape(named)}</span>' if named else ""
    return (
        f'<section class="walk" id="{escape(scenario.anchor, quote=True)}">'
        f'<div class="head"><h2>{write.text(scenario.name)}</h2>'
        f'<span class="muted">{escape(summary)}</span></div>'
        f"{identity}{why}{steps}{_checks(write, scenario.checks)}</section>"
    )


def _rail(write: Markup, scenarios: Sequence[ScenarioView]) -> str:
    """Navigation between scenarios, where there is more than one.

    One scenario has nothing to navigate between. The rail would repeat
    the heading below it and take width from every step row.
    """
    if len(scenarios) < 2:
        return ""
    links = "".join(
        f'<a href="#{escape(scenario.anchor, quote=True)}">'
        f'<span class="dot d-{scenario.tone}"></span>'
        f'<span class="what">{write.text(scenario.name)}</span>'
        f'<span class="count">{escape(scenario.status)}'
        f" · {len(scenario.steps)} steps</span></a>"
        for scenario in scenarios
    )
    return f'<nav class="rail" aria-label="Scenarios"><h2>Scenarios</h2>{links}</nav>'


def _verdict(write: Markup, view: ReportView) -> str:
    verdict = view.verdict
    jump = (
        f'<a class="jump" href="#{escape(verdict.anchor, quote=True)}">'
        f"{escape(verdict.toward)}</a>"
        if verdict.anchor and verdict.toward
        else ""
    )
    return (
        f'<section class="verdict t-{verdict.tone}" aria-label="Verdict">'
        f'<div class="said"><div class="line">'
        f'<span class="badge b-{verdict.tone}">{escape(verdict.badge)}</span> '
        f'<span class="muted">{escape(verdict.counts)}</span></div>'
        f"<h2>{write.named(verdict.headline)}</h2>"
        f"<p>{write.named(verdict.detail)}</p></div>{jump}</section>"
    )


def _tiles(view: ReportView) -> str:
    tiles = "".join(
        f'<div class="tile"><span class="label">{escape(tile.label)}</span>'
        f'<span class="value">{escape(tile.value)}</span>'
        + (f'<span class="note">{escape(tile.note)}</span>' if tile.note else "")
        + "</div>"
        for tile in view.tiles
    )
    return f'<section class="tiles" aria-label="Run summary">{tiles}</section>'


def _journey(write: Markup, view: ReportView) -> str:
    secrets = (
        f'<span class="secrets">{theme.MARKS['lock']}{view.secrets} '
        f'{"value" if view.secrets == 1 else "values"} hidden in this report</span>'
        if view.secrets
        else ""
    )
    return (
        '<section class="panel" aria-label="Journey">'
        f'<div class="head"><h2>Test Flow</h2>{secrets}</div>'
        f"<p>{write.named(view.description)}</p></section>"
    )


def _findings(write: Markup, findings: Sequence[FindingView]) -> str:
    if not findings:
        return ""
    rows = "".join(
        f'<div class="check"><span class="badge b-{finding.tone}">{escape(finding.severity)}</span>'
        f'<span class="rule">{write.text(finding.description)}</span>'
        f'<span class="said">{escape(finding.kind)}'
        + (f" · {_reach(finding)}" if _reach(finding) else "")
        + (f" · {write.text(finding.where)}" if finding.where else "")
        + (
            f' · <a href="#{escape(finding.anchor, quote=True)}">View evidence</a>'
            if finding.anchor
            else ""
        )
        + "</span></div>"
        for finding in findings
    )
    return (
        '<section class="rows" id="findings"><div class="head"><h2>Observed Issues</h2>'
        '<span class="muted">Additional issues observed during the run</span>'
        f"</div>{rows}</section>"
    )


def _reach(finding: FindingView) -> str:
    """How far one issue went, where it went further than one screen."""
    if finding.seen < 2:
        return escape(finding.screens[0]) if finding.screens else ""
    named = ", ".join(escape(screen) for screen in finding.screens[:4])
    more = "…" if len(finding.screens) > 4 else ""
    return f"seen {finding.seen} times on {named}{more}"


def _unevidenced(write: Markup, rules: Sequence[str]) -> str:
    if not rules:
        return ""
    rows = "".join(
        f'<div class="check"><span class="badge b-warn">Not evidenced</span>'
        f'<span class="rule">{write.named(rule)}</span></div>'
        for rule in rules
    )
    return (
        '<section class="rows"><div class="head"><h2>Security Rules Not Evaluated</h2>'
        '<span class="muted">No scenario reached a screen where Yewo could evaluate these rules.</span>'
        f"</div>{rows}</section>"
    )


def _cloud(view: ReportView) -> str:
    if not OFFER_CLOUD or not view.replayable:
        return ""
    lead = escape(CLOUD_LEAD.format(count=view.replayable))
    return (
        '<section class="cloud" aria-label="Yewo Cloud">'
        f'<div class="said"><span class="lead">{lead}</span>'
        f'<span class="muted">{escape(CLOUD_BODY)}</span></div>'
        f'<a href="{CLOUD}">See Yewo Cloud</a></section>'
    )


def _footer(view: ReportView) -> str:
    beside = (
        "Saved with this run: "
        + " · ".join(
            f'<a class="mono" href="../{escape(name, quote=True)}">{escape(name)}</a>'
            for name in view.artifacts
        )
        if view.artifacts
        else "This file works offline and can be attached to a CI run."
    )
    return f"<footer><span>{beside}</span></footer>"


def document(view: ReportView, scrubber: Optional[SecretScrubber] = None) -> str:
    """The whole page, as one string."""
    write = Markup(scrubber)
    body = "".join(
        [
            '<header class="top"><div class="bar"><div class="brand">'
            f'<span class="logo">{theme.MARKS['logo']}<span class="name">yewo</span></span>'
            '<span class="slash" aria-hidden="true">/</span>'
            '<span class="what">Run report</span></div>'
            f'<div class="stamp"><span>Generated {escape(view.generated)} · '
            f"yewo {escape(view.version)}</span>{theme.MARKS['toggle']}</div></div></header>",
            "<main>",
            f'<div class="title"><span class="muted">Journey · '
            f'<span class="mono">{write.text(view.subject)}</span></span>'
            f"<h1>{write.text(view.title)}</h1></div>",
            _verdict(write, view),
            _tiles(view),
            _journey(write, view),
            '<div class="split">',
            _rail(write, view.scenarios),
            '<div class="walks">',
            "".join(_walk(write, scenario) for scenario in view.scenarios),
            "</div></div>",
            _findings(write, view.findings),
            _unevidenced(write, view.unevidenced),
            _cloud(view),
            _footer(view),
            "</main>",
        ]
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Yewo run report · {write.text(view.title)}</title>\n"
        f"<style>{theme.stylesheet()}</style>\n</head>\n<body>\n{body}\n"
        f"<script>{theme.script()}</script>\n</body>\n</html>\n"
    )


def _screenshots_href(output_dir: Path, screenshots: Optional[Path]) -> str:
    """Where the images are, written relative to wherever this report lands."""
    if screenshots is None:
        return "screenshots"
    try:
        return os.path.relpath(screenshots, output_dir).replace("\\", "/")
    except ValueError:
        # A different drive on Windows has no relative path. An absolute one
        # still opens on the machine that produced the run.
        return screenshots.as_uri()


def save_html_report(
    report: JourneyReport,
    output_dir: Path,
    screenshots: Optional[Path] = None,
    scrubber: Optional[SecretScrubber] = None,
    label: Optional[str] = None,
    usage: Optional[dict] = None,
    artifacts: Iterable[str] = (),
) -> Path:
    """Writes the run report a person reads, beside the ones a machine reads.

    `screenshots` is where this run's images went. The report links them
    relative to itself, so moving the directory moves the report with them.
    """
    view = page_of(
        report,
        generated=now(),
        # The run's own version, so a page says what produced the run rather
        # than what rendered it. They differ the moment a record is re-read.
        version=report.environment.yewo_version if report.environment else __version__,
        usage=usage,
        screenshots_href=_screenshots_href(output_dir, screenshots),
        artifacts=tuple(artifacts),
    )
    return save_page(view, output_dir, label=label, scrubber=scrubber)


def now() -> str:
    """When this page was written, in the words the page shows it in."""
    return time.strftime("%d %b %Y, %H:%M UTC", time.gmtime())


def save_page(
    view: ReportView,
    output_dir: Path,
    label: Optional[str] = None,
    scrubber: Optional[SecretScrubber] = None,
) -> Path:
    """Writes one page, whichever supplier built it.

    The run that is finishing and the reader who opened `run.json` arrive
    here with the same values. Neither can produce a page the other could
    not.
    """
    stamp = label or time.strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"report_{safe_artifact_stem(view.subject)}_{stamp}.html"
    write_atomic(filepath, document(view, scrubber))
    logger.info("HTML test report successfully written to %s", filepath)
    return filepath
