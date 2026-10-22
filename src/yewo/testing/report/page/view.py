"""The facts a run left, arranged as the page that reports them.

This module decides what a reader is shown, and in what order. The verdict
comes first. Then what the run cost. Then the scenario that failed, and the
step it failed on. The renderer beside it turns these values into markup and
decides nothing. So what the page says can be tested without parsing HTML.

Every string here has already been through the scrubber that built the report.
The renderer escapes them. This module does not. A view value is text, and not
markup.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from yewo.testing.judge import rules, verdicts
from yewo.testing.models import QAIssue

from .facts import Tile, seconds, tiles_of
from .walk import StepView, anchor_of, failed_at, step_views
from .source import RunSource, ScenarioSource

TONES: Dict[str, str] = {
    "PASSED": "pass",
    "FAILED": "fail",
    "UNVERIFIED": "warn",
    "UNSUPPORTED": "warn",
    "ERROR": "warn",
    "ABORTED": "warn",
    "SKIPPED": "muted",
    "PENDING": "muted",
    "RUNNING": "muted",
}
"""How each verdict is coloured.

Red is kept for what the app did. A scenario that ended on the provider, the
device or missing evidence is amber. Nothing was learned about the app. A
reader of a red build must not go looking in it.
"""

VERBS: Dict[str, str] = {
    "FAILED": "failed",
    "UNVERIFIED": "reached no verdict",
    "UNSUPPORTED": "could not be run here",
    "ERROR": "could not run",
    "ABORTED": "was interrupted",
    "SKIPPED": "was skipped",
    "PENDING": "has not started",
    "RUNNING": "is still running",
}
"""What the headline says a scenario did, in the reader's words."""

SEVERITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}
"""Worst first, because a findings list is read from the top."""

SEVERITY_TONES = {"critical": "fail", "high": "fail", "medium": "warn", "low": "muted"}

RUN_FILES = ("screenshots/", "logs/", "run.json")
"""What else a run leaves, named in the page's footer.

Not `ARTIFACTS`, which `sensitivity` already uses for the classes that say
what may be done with each kind of file. These are paths a reader opens.

Here rather than beside the writer, because both suppliers of a page name
the same files. A reader who opened the record is standing in the directory
that holds them.
"""


SECRET = re.compile(r"\{\{([A-Za-z0-9_.-]+)\}\}")
"""A value the scrubber masked, left in the text under the name that declared it."""


SCOPE_WRITTEN_IN = re.compile(r"^\s*\w+:\s*(?:true|false)\s*[—-]\s*")
"""The scope, where the generator wrote it into the statement as well.

`ExpectedOutcome` carries the scope as a field, so a statement that opens
with `current_screen: true — ` is saying it twice. A reader is shown the
requirement. The generator writing it at all is recorded as debt.
"""


@dataclass(frozen=True)
class CheckView:
    """One rule, as the judge settled it.

    Not one row per answer. The auditor answers every rule on every screen.
    A rule about the order screen is answered on the cart too, where the
    honest answer is "no". Rendering those as results turns a scenario that
    passed into a page of failures. The verdict took one answer per rule,
    and that is the one shown.
    """

    rule: str
    tone: str
    status: str
    reasoning: str
    named: str = ""
    screen: str = ""
    asked: int = 0


@dataclass(frozen=True)
class FindingView:
    """One issue, however many times a run saw it.

    The auditor reports per screen, so one debug ribbon becomes a row per
    screen it was visible on. Gathered by kind, a reader is shown the
    defect and how far it reaches.
    """

    severity: str
    tone: str
    kind: str
    description: str
    where: str
    seen: int = 1
    screens: Tuple[str, ...] = ()
    anchor: str = ""


@dataclass(frozen=True)
class ScenarioView:
    """One scenario: its verdict, its walk and what it was judged on."""

    anchor: str
    name: str
    test_id: str
    cell: str
    tone: str
    status: str
    summary: str
    failure_reason: str = ""
    steps: Tuple[StepView, ...] = ()
    checks: Tuple[CheckView, ...] = ()


@dataclass(frozen=True)
class VerdictView:
    """What the run came to, in the first thing the page shows."""

    tone: str
    badge: str
    counts: str
    headline: str
    detail: str
    anchor: str = ""
    toward: str = ""


@dataclass(frozen=True)
class ReportView:
    """The whole page, as values."""

    title: str
    subject: str
    generated: str
    version: str
    verdict: VerdictView
    tiles: Tuple[Tile, ...]
    description: str
    secrets: int
    scenarios: Tuple[ScenarioView, ...]
    findings: Tuple[FindingView, ...] = ()
    unevidenced: Tuple[str, ...] = ()
    artifacts: Tuple[str, ...] = ()
    replayable: int = 0


def _declared(scenario: ScenarioSource) -> Dict[str, str]:
    """The rules as the journey wrote them, by id.

    A verification holds the rule as the auditor echoed it, and an auditor
    echoes differently each time. The journey's own words are the same on
    every row.
    """
    given = scenario.verdict_inputs
    if given is None:
        return {}
    written = {
        rules.rule_id(rules.EXPECTED, index): SCOPE_WRITTEN_IN.sub("", outcome.statement)
        for index, outcome in enumerate(given.expected_outcomes)
    }
    written.update(
        {
            rules.rule_id(rules.SECURITY, index): rule
            for index, rule in enumerate(given.security_rules)
        }
    )
    return written


def _named(rule_id: str) -> str:
    """What a reader calls the rule, from the id the prompt gave it.

    Counted from one. The id is stored from zero because it indexes a
    list. A reader asked to look at "expected outcome 0" is being shown
    an array offset.
    """
    kind, _, index = rule_id.partition(":")
    spelled = {rules.EXPECTED: "Expected outcome", rules.SECURITY: "Security rule"}
    if not index.isdigit():
        return rule_id
    return f"{spelled.get(kind, kind.capitalize())} {int(index) + 1}"


def _checks(scenario: ScenarioSource) -> Tuple[CheckView, ...]:
    """The rules this scenario was judged on, failures first."""
    verifications = list(scenario.rule_verifications)
    declared = _declared(scenario)
    asked: Dict[str, int] = {}
    for verification in verifications:
        if verification.rule_id:
            asked[verification.rule_id] = asked.get(verification.rule_id, 0) + 1
    settled = [
        CheckView(
            rule=declared.get(rule_id) or verification.rule,
            tone="pass" if verification.status == "passed" else "fail",
            status="Passed" if verification.status == "passed" else "Failed",
            reasoning=verification.reasoning,
            named=_named(rule_id),
            screen=verification.screen_name or "",
            asked=asked.get(rule_id, 0),
        )
        for rule_id, verification in verdicts.per_rule(verifications).items()
    ]
    return tuple(sorted(settled, key=lambda check: check.tone == "pass"))


def _summary(scenario: ScenarioSource) -> str:
    """The line under a scenario's name: how far it got, and for how long."""
    parts = []
    if scenario.steps_executed:
        parts.append(f"{scenario.steps_executed} steps")
    if scenario.duration:
        parts.append(seconds(scenario.duration))
    if scenario.cause:
        parts.append(scenario.cause)
    return " · ".join(parts)


def _scenarios(report: RunSource, href: str) -> Tuple[ScenarioView, ...]:
    views = []
    for scenario in report.scenarios:
        tone = TONES.get(scenario.status, "warn")
        opened = failed_at(scenario.steps) if tone != "pass" else None
        views.append(
            ScenarioView(
                anchor=anchor_of(scenario.test_id),
                name=scenario.scenario_name,
                test_id=scenario.test_id,
                cell=scenario.cell,
                tone=tone,
                status=scenario.status.title(),
                summary=_summary(scenario),
                failure_reason=scenario.failure_reason or "",
                steps=step_views(scenario, href, opened),
                checks=_checks(scenario),
            )
        )
    return tuple(views)


def _same_issue(issue: QAIssue, scenario_name: str) -> Tuple[str, str, str, str]:
    """What makes two reports the same defect rather than the same category.

    The kind, the screen, the severity and the words. Nothing less is
    safe. A category is not an identity. Two unrelated overflows on two
    screens share `layout_overflow`. Gathering them invents one finding
    with the first one's severity and the second one's words.

    What this does catch is the real duplicate. The auditor is asked on
    arrival and on departure, so one defect on one screen is reported
    twice in the same words.
    """
    return (
        issue.issue_type.value,
        issue.current_screen or "",
        issue.severity,
        (issue.description or "").strip(),
    )


def _where_seen(scenario: ScenarioSource, issue: QAIssue) -> str:
    """The step this issue can be looked at, where the run recorded one.

    Nothing is guessed. Matching on the screen name finds the first visit
    to a screen. A run reaches one screen many times, so a defect seen on
    the second visit would link to the first. A link to the wrong
    screenshot is worse than no link. It shows a reader an image that does
    not hold the defect, and invites them to doubt the finding.
    """
    if issue.step is None:
        return ""
    for step in scenario.steps:
        if step.n == issue.step:
            return f"{anchor_of(scenario.test_id)}-step-{step.n}"
    return ""


def _findings(report: RunSource) -> Tuple[FindingView, ...]:
    """Every issue a run saw, worst first, with exact repeats gathered."""
    seen: Dict[Tuple[str, str, str, str], List[Tuple[QAIssue, str]]] = {}
    toward: Dict[Tuple[str, str, str, str], str] = {}
    for scenario in report.scenarios:
        for issue in scenario.detected_issues:
            key = _same_issue(issue, scenario.scenario_name)
            seen.setdefault(key, []).append((issue, scenario.scenario_name))
            toward.setdefault(key, _where_seen(scenario, issue))
    one_scenario = len(report.scenarios) < 2
    found = [
        _gathered(rows, one_scenario, toward[key]) for key, rows in seen.items()
    ]
    return tuple(sorted(found, key=lambda one: SEVERITY.get(one.severity.lower(), 9)))


def _gathered(
    rows: List[Tuple[QAIssue, str]], one_scenario: bool, anchor: str = ""
) -> FindingView:
    """One defect, however many times it was reported in the same words."""
    issue = rows[0][0]
    screens = tuple(
        dict.fromkeys(one.current_screen for one, _ in rows if one.current_screen)
    )
    named = "" if one_scenario else " · ".join(dict.fromkeys(name for _, name in rows))
    return FindingView(
        severity=issue.severity.title(),
        tone=SEVERITY_TONES.get(issue.severity, "warn"),
        kind=issue.issue_type.value.replace("_", " ").title(),
        description=issue.description,
        where=named,
        seen=len(rows),
        screens=screens,
        anchor=anchor,
    )


UNFINISHED = ("PENDING", "RUNNING")
"""What a scenario is called before it reports anything."""


def _observed(findings: int, unevidenced: int) -> str:
    """What a passing run still has to say about what it saw.

    A run can reach every outcome it was given and still have watched the
    app render a card number in plain text. It can leave a security rule
    no screen could answer. "Nothing to report" above a page that lists
    either one is the page contradicting itself.
    """
    notes = []
    if findings:
        notes.append(f"{findings} issue{'' if findings == 1 else 's'} observed")
    if unevidenced:
        notes.append(
            f"{unevidenced} security rule{'' if unevidenced == 1 else 's'} not evaluated"
        )
    if not notes:
        return "All expected outcomes met. No issues observed."
    return f"All expected outcomes met. {' · '.join(notes).capitalize()}."


def _every_scenario_passed(report: RunSource) -> bool:
    """Whether the rows support the summary above them.

    Read from the scenarios rather than from the run's own word. A record
    is a file, and a run that says PASSED over a failed row is a record
    that disagrees with itself. The rows are the evidence.
    """
    return bool(report.scenarios) and all(
        one.status == "PASSED" for one in report.scenarios
    )


def _verdict(
    report: RunSource,
    scenarios: Sequence[ScenarioView],
    findings: int = 0,
    unevidenced: int = 0,
) -> VerdictView:
    """What the page leads with, which is the first thing that went wrong."""
    total = len(report.scenarios)
    counts = f"{report.passed_count} of {total} scenarios passed"
    if report.status == "PASSED" and _every_scenario_passed(report):
        return VerdictView(
            tone="pass",
            badge="Passed",
            counts=counts,
            headline=f"Every scenario in {report.journey_name} passed.",
            detail=_observed(findings, unevidenced),
            anchor="findings" if findings else "",
            toward="View issues" if findings else "",
        )
    unsuccessful = next(
        (view for view in scenarios if view.tone in ("fail", "warn")), None
    )
    if unsuccessful is None:
        # A record is written before the first scenario starts and updated
        # as each one ends, so a reader can open it while the run is going.
        # Nothing has gone wrong there. Nothing has happened yet.
        if report.status == "RUNNING":
            # By status, not by tone: a skipped scenario is finished and muted.
            done = sum(
                1 for one in report.scenarios if one.status not in UNFINISHED
            )
            return VerdictView(
                tone="muted",
                badge="Running",
                counts=counts,
                headline=f"{report.journey_name} is still running.",
                detail=f"{done} of {total} scenarios have finished.",
            )
        if report.status == "PASSED":
            return VerdictView(
                tone="warn",
                badge="Passed",
                counts=counts,
                headline=f"{report.journey_name} reported no failure.",
                detail=_observed(findings, unevidenced),
                anchor="findings" if findings else "",
                toward="View issues" if findings else "",
            )
        return VerdictView(
            tone="fail",
            badge="Failed",
            counts=counts,
            headline=f"{report.journey_name} did not finish.",
            detail="No scenario reported a verdict of its own.",
        )
    failed_at = next(
        (step for step in unsuccessful.steps if step.tone == "fail"), None
    )
    where = f" at step {int(failed_at.number)}" if failed_at else ""
    status = next(
        (one.status for one in report.scenarios if one.test_id == unsuccessful.test_id),
        "FAILED",
    )
    return VerdictView(
        tone=unsuccessful.tone,
        badge="Running" if report.status == "RUNNING" else unsuccessful.status,
        counts=counts,
        headline=f"“{unsuccessful.name}” {VERBS.get(status, 'failed')}{where}.",
        detail=unsuccessful.failure_reason or "The run recorded no reason.",
        anchor=failed_at.anchor if failed_at else unsuccessful.anchor,
        toward="Jump to the failure",
    )


def _secrets(report: RunSource, scenarios: Sequence[ScenarioView]) -> int:
    """How many declared values this page shows by name instead of by value."""
    texts = [report.description]
    for view in scenarios:
        texts.append(view.failure_reason)
        for step in view.steps:
            texts.append(step.title)
            texts.extend(note.body for note in step.notes)
        texts.extend(check.reasoning for check in view.checks)
        texts.extend(check.rule for check in view.checks)
    found = {name for text in texts for name in SECRET.findall(text or "")}
    return len(found)


def _replayable(report: RunSource) -> int:
    """Steps that worked, which are the ones a recorded path could repeat."""
    return sum(
        1
        for scenario in report.scenarios
        for step in scenario.steps
        if step.outcome is not None and step.outcome.success
    )


def page_of(
    report: RunSource,
    generated: str,
    version: str,
    usage: Optional[dict] = None,
    screenshots_href: str = "screenshots",
    artifacts: Sequence[str] = (),
) -> ReportView:
    """The whole page for one journey report, as values the renderer writes out."""
    scenarios = _scenarios(report, screenshots_href.rstrip("/"))
    found = _findings(report)
    unevidenced = tuple(report.unevidenced_security_rules)
    return ReportView(
        title=report.journey_name,
        subject=report.journey_id,
        generated=generated,
        version=version,
        verdict=_verdict(report, scenarios, len(found), len(unevidenced)),
        tiles=tiles_of(report, usage),
        description=report.description,
        secrets=_secrets(report, scenarios),
        scenarios=scenarios,
        findings=found,
        unevidenced=unevidenced,
        artifacts=tuple(artifacts),
        replayable=_replayable(report),
    )
