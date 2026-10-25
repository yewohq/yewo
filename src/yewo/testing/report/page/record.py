"""The page, read back out of `run.json`.

The other supplier of a page is a run that is still in memory. This one is a
file. Whoever downloaded it opens it: a CI job attaching the report to a pull
request, or a person reading last week's failure. Nothing here needs the run
that wrote the file. It needs no device and no model.

That is what makes the record canonical rather than one output among four. A
page built this way cannot say what the record does not hold. What it cannot
say is a gap in the record, and not a gap in the page.

Every value arrives as JSON somebody else wrote. The whole file goes
through a model before the page reads any of it. Coercing values one at a
time lets a bad one through as a plausible page. A status nothing
recognises renders as a verdict. A duration that is a word escapes as
whatever `float` raises.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from yewo.agent.run.record import StepRecord
from yewo.errors import UsageError
from yewo.testing.models import (
    Measurement,
    PolicyCheckReport,
    QAIssue,
    REPORT_STATUS_ALIASES,
    ReportStatus,
    RuleVerification,
    RunEnvironment,
    VerdictInputs,
)
from yewo.testing.report.manifest import SCHEMA
from yewo.testing.lifecycle import LifecycleOperation

from .view import RUN_FILES, ReportView, page_of

MAX_DISPLAYED_ERRORS = 3
"""How many of a record's errors are named. The first is usually the one."""

Number = Union[StrictFloat, StrictInt]
"""A JSON number, and not a string that looks like one.

The record is written by Yewo and read here. A quoted number means the file
was edited, or assembled by something else. A reader that accepted it would
be guessing at what it meant.
"""


class _RecordedSubject(BaseModel):
    """What the run was about, as the record holds it."""

    model_config = ConfigDict(extra="ignore")

    kind: str = "journey"
    id: str = "run"
    name: str = ""
    description: str = ""


class _RecordedScenarioEntry(BaseModel):
    """One scenario of a record, read rather than coerced.

    `extra="ignore"` on purpose. The record carries more per scenario than a
    page shows. A reader that refused a file over a field it does not render
    would refuse to show a run it could show.
    """

    model_config = ConfigDict(extra="ignore")

    test_id: str = ""
    name: str = ""
    cell: str = ""
    status: str = ""
    cause: Optional[str] = None
    duration_seconds: Optional[Number] = None
    steps_executed: StrictInt = 0
    failure_reason: Optional[str] = None
    steps: List[StepRecord] = Field(default_factory=list)
    rule_verifications: List[RuleVerification] = Field(default_factory=list)
    detected_issues: List[QAIssue] = Field(default_factory=list)
    lifecycle_operations: List[LifecycleOperation] = Field(default_factory=list)
    measurements: List[Measurement] = Field(default_factory=list)
    verdict_inputs: Optional[VerdictInputs] = None


class _RecordedUsage(BaseModel):
    """What the run spent. Only the fields the band reads."""

    model_config = ConfigDict(extra="ignore")

    total_calls: StrictInt = 0
    total_estimated_cost_usd: Optional[Number] = None
    cost_unknown_calls: StrictInt = 0


class _RunRecordPayload(BaseModel):
    """`run.json`, as much of it as a page reads.

    `counts` is not read. The scenarios are in the file, so counting them
    here keeps one owner for how many passed. A record whose summary and
    whose rows disagree would otherwise render a green headline over a list
    of failures.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: StrictInt = Field(alias="schema")
    subject: _RecordedSubject = Field(default_factory=_RecordedSubject)
    status: str = ""
    duration_seconds: Optional[Number] = None
    scenarios: List[_RecordedScenarioEntry]
    policy_checks: List[PolicyCheckReport] = Field(default_factory=list)
    unevidenced_security_rules: List[str] = Field(default_factory=list)
    identity: Optional[RunEnvironment] = None
    usage: Optional[_RecordedUsage] = None


@dataclass(frozen=True)
class RecordedScenario:
    """One scenario of a record, under the names the page reads."""

    test_id: str
    scenario_name: str
    cell: str
    status: ReportStatus
    cause: Optional[str]
    duration: float
    steps_executed: int
    failure_reason: Optional[str]
    steps: Sequence[StepRecord] = ()
    rule_verifications: Sequence[RuleVerification] = ()
    detected_issues: Sequence[QAIssue] = ()
    lifecycle_operations: Sequence[LifecycleOperation] = ()
    measurements: Sequence[Measurement] = ()
    verdict_inputs: Optional[VerdictInputs] = None


@dataclass(frozen=True)
class RecordedRun:
    """One record, under the names the page reads."""

    journey_id: str
    journey_name: str
    description: str
    status: ReportStatus
    total_duration: float
    passed_count: int
    skipped_count: int
    scenarios: Sequence[RecordedScenario] = ()
    policy_checks: Sequence[PolicyCheckReport] = ()
    unevidenced_security_rules: Sequence[str] = ()
    environment: Optional[RunEnvironment] = None


def _format_validation_errors(exc: ValidationError) -> str:
    """A validation report, as the fields it is about and what was wrong."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()[:MAX_DISPLAYED_ERRORS]
    )


def _normalize_status(status: str, field_context: str) -> ReportStatus:
    """A status the rest of the page can act on, or a refusal.

    An unrecognised word would reach the page as a verdict. The renderer
    titles whatever it is given, and colours what it does not know amber. A
    reader would then be told a scenario was "Teleported".
    """
    raw_status = (status or "").strip()
    aliased_status = REPORT_STATUS_ALIASES.get(raw_status.lower())
    if aliased_status is not None:
        return aliased_status
    try:
        return ReportStatus(raw_status.upper())
    except ValueError:
        raise UsageError(
            f"The record's {field_context} is not a status Yewo writes: "
            f"{raw_status!r}"
        ) from None


def _scenario(entry: _RecordedScenarioEntry) -> RecordedScenario:
    return RecordedScenario(
        test_id=entry.test_id,
        scenario_name=entry.name,
        cell=entry.cell,
        status=_normalize_status(
            entry.status, f"status for {entry.test_id or 'a scenario'}"
        ),
        cause=entry.cause,
        duration=entry.duration_seconds or 0.0,
        steps_executed=entry.steps_executed,
        failure_reason=entry.failure_reason,
        steps=tuple(entry.steps),
        rule_verifications=tuple(entry.rule_verifications),
        detected_issues=tuple(entry.detected_issues),
        lifecycle_operations=tuple(entry.lifecycle_operations),
        measurements=tuple(entry.measurements),
        verdict_inputs=entry.verdict_inputs,
    )


def run_of(payload: Mapping[str, Any]) -> RecordedRun:
    """A record, as the run the page reads.

    Raises:
        UsageError: The payload is not a run record, or holds a value the
            model that owns it refuses.
    """
    if not isinstance(payload, Mapping) or "scenarios" not in payload:
        raise UsageError("That file does not hold a run record.")
    try:
        read = _RunRecordPayload.model_validate(dict(payload))
    except ValidationError as exc:
        raise UsageError(
            f"That run record is not readable: {_format_validation_errors(exc)}"
        ) from exc
    if read.subject.kind != "journey":
        raise UsageError(
            f"That record is a {read.subject.kind} run, and this page reports "
            "on journeys. Its walk is not a scenario and is not shown here."
        )
    if read.schema_version != SCHEMA:
        raise UsageError(
            f"That run record is schema {read.schema_version}, and this Yewo "
            f"reads schema {SCHEMA}. Nothing translates between them."
        )

    scenarios = tuple(_scenario(entry) for entry in read.scenarios)
    return RecordedRun(
        journey_id=read.subject.id,
        journey_name=read.subject.name,
        description=read.subject.description,
        status=_normalize_status(read.status, "run status"),
        total_duration=read.duration_seconds or 0.0,
        passed_count=sum(
            1 for scenario in scenarios if scenario.status == ReportStatus.PASSED
        ),
        skipped_count=sum(
            1 for scenario in scenarios if scenario.status == ReportStatus.SKIPPED
        ),
        scenarios=scenarios,
        policy_checks=tuple(read.policy_checks),
        unevidenced_security_rules=tuple(read.unevidenced_security_rules),
        environment=read.identity,
    )


def page_of_record(
    payload: Mapping[str, Any],
    generated: str,
    screenshots_href: str = "screenshots",
) -> ReportView:
    """The whole page for one record, as values the renderer writes out.

    The version shown is the one in the record. A page rendered today from
    an old run is about that run. Stamping it with whatever is installed
    here would attribute the run to code that did not produce it.

    The footer names the files beside the record, which is the list the run
    itself named. The record's own `artifacts` are sensitivity classes
    rather than file names, and they answer a different question.

    Run-level policy evaluations are parsed with the rest of the record. They
    are rendered separately from scenario evaluations.
    """
    run = run_of(payload)
    usage_summary = _RunRecordPayload.model_validate(dict(payload)).usage
    return page_of(
        run,
        generated=generated,
        version=run.environment.yewo_version if run.environment else "",
        usage=usage_summary.model_dump() if usage_summary else None,
        screenshots_href=screenshots_href,
        artifacts=RUN_FILES,
    )
