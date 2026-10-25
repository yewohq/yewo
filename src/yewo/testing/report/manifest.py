"""`run.json` — one record per run, and what every report is a view over.

This is the file that answers "what happened", against the lock's "what is
the test". It lists every scenario before the first one starts, and updates
each as it finishes. So it can be read, or tailed, while the run is going.

That is what the scenario plan did. This is that same writer, grown into the
whole record rather than joined by a second file. A second writer of a run's
outcome would be the fifth, and they could disagree.

Four things it carries that no earlier artifact did.

- **`schema`**, read before anything else in the file.
- **Identity**: the build, the target, the commit and the job.
- **Inputs**: the lock and the configuration this run was given, as hashes.
- **Per-step records**, so a reader can ask which step, and not only how many.

Every string in it is scrubbed. It is an artifact, so it is uploaded,
attached to a review and kept.
"""

import json
import logging
from datetime import datetime
from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from yewo.agent.run.record import CallRow, StepRecord
from yewo.identity import new_uuid
from yewo.testing.models import (
    GeneratedTestCase,
    JourneyStatus,
    Measurement,
    QAIssue,
    PolicyCheckReport,
    ReportStatus,
    RuleVerification,
    RunEnvironment,
    ScenarioReport,
    VerdictInputs,
)
from yewo.policy import policy_checks_succeeded
from yewo.utils.secrets import SecretScrubber
from yewo.testing.lifecycle import LifecycleOperation

logger = logging.getLogger(__name__)

SCHEMA = 4
"""The version of this file's shape.

4 adds direct measurement samples to each scenario. Schema 3 added lifecycle
operations. Schema 2 added `verdict_inputs`.
A record written under 1 holds too little to reach its verdict again.
"""

ABORTED = ReportStatus.ABORTED
"""A run that was interrupted. It never reached a verdict of its own."""

UNCOVERED = (ReportStatus.UNSUPPORTED, ReportStatus.ERROR)
"""Statuses that mean no verdict was reached about the app."""


def _is_uncovered(entry: "ScenarioEntry") -> bool:
    """Whether a row describes a cell that produced no runnable scenario."""
    return entry.status in UNCOVERED or (
        entry.status == ReportStatus.UNVERIFIED and not entry.category
    )


def _now() -> str:
    """One spelling of the timestamp every row and the file itself carry."""
    return datetime.now().isoformat(timespec="seconds")


class ScenarioEntry(BaseModel):
    """One scenario's row, from listed to finished.

    A field a scenario only has once it has run stays None until then, and
    the writer leaves it out. So a row grows as the run learns. The fields a
    row can hold are fixed here, rather than accruing wherever a status is
    set.
    """

    position: int
    scenario_uuid: str
    test_id: str
    name: str
    cell: str
    category: str
    scenario_type: str
    priority: str
    steps_planned: int
    steps_instructions: List[str]
    expected_outcomes: List[str]
    required_metrics: List[str] = Field(default_factory=list)
    outcome_uuids: List[str] = Field(
        default_factory=list,
        description="One per expected outcome, in the same order.",
    )
    status: ReportStatus = ReportStatus.PENDING
    cause: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    steps_executed: Optional[int] = None
    duration_seconds: Optional[float] = None
    failure_reason: Optional[str] = None
    issues_found: Optional[int] = None
    steps: List[StepRecord] = Field(default_factory=list)
    calls: List[CallRow] = Field(
        default_factory=list,
        description=(
            "Calls this scenario paid for outside any step. The judges that "
            "answer once the walk is over are here, and the rest are under "
            "the step that made them."
        ),
    )
    rule_verifications: List[RuleVerification] = Field(default_factory=list)
    detected_issues: List[QAIssue] = Field(default_factory=list)
    lifecycle_operations: List[LifecycleOperation] = Field(default_factory=list)
    measurements: List[Measurement] = Field(default_factory=list)
    verdict_inputs: Optional[VerdictInputs] = Field(
        default=None,
        description=(
            "What the verdict was decided from. Absent on a row Yewo wrote "
            "about itself, and on a scenario that never reached a judge."
        ),
    )


class RunSubject(BaseModel):
    """What the run was about: a journey, or one plain-English goal."""

    kind: Literal["journey", "goal"]
    id: str
    name: str
    description: str = Field(
        default="",
        description=(
            "The journey in the words it was written in. The record holds it "
            "so a reader of the file alone can tell what was asked for."
        ),
    )


class RunInputs(BaseModel):
    """What this run was given, as hashes, so two runs can be told apart."""

    subject: str
    subject_sha256: str
    coverage_declared: List[str] = Field(default_factory=list)
    lock_sha256: Optional[str] = None
    scenarios_from_lock: int = 0
    scenarios_generated: int = 0


class FailureSummary(BaseModel):
    """Where to look first, so a reader does not scroll a passing run."""

    scenario_uuid: str
    test_id: str
    rule_id: Optional[str] = None
    step: Optional[int] = None


class RunManifest:
    """Writes the run's record and keeps it current as the run goes."""

    def __init__(
        self,
        path,
        subject: RunSubject,
        scenarios: Sequence[GeneratedTestCase],
        environment: RunEnvironment,
        inputs: RunInputs,
        scrubber: Optional[SecretScrubber] = None,
        mode: str = "test",
        artifacts: tuple = (),
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.subject = subject
        self.environment = environment
        self.inputs = inputs
        self.mode = mode
        self.artifacts = tuple(artifacts)
        self.run_uuid = new_uuid()
        self.started_at = datetime.now()
        self.status = ReportStatus.RUNNING
        self.unevidenced_security_rules: List[str] = []
        self.policy_checks: List[PolicyCheckReport] = []
        self.duration: Optional[float] = None
        self.usage: Optional[dict] = None
        self.steps: List[StepRecord] = []
        """A goal run has no scenarios, so its steps sit at the top level."""

        self.unwritten: Optional[OSError] = None
        """The last write failure, or None. Read at the end of the run."""

        self.calls: List[CallRow] = []
        """Calls that belong to the run rather than to any scenario in it.

        Generation is the one that matters: it happens once, before the first
        scenario, and costs real money. Left out, the calls recorded in this
        file would not add up to what `usage` says the run spent.
        """
        self._scrubber = scrubber
        self.entries: dict[str, ScenarioEntry] = {
            scenario.test_id: self._listed(index + 1, scenario)
            for index, scenario in enumerate(scenarios)
        }

    def _listed(self, position: int, scenario: GeneratedTestCase) -> ScenarioEntry:
        return ScenarioEntry(
            position=position,
            scenario_uuid=scenario.scenario_uuid,
            test_id=scenario.test_id,
            name=self._scrub(scenario.scenario_name),
            cell=scenario.cell,
            category=scenario.category.value,
            scenario_type=scenario.scenario_type.value,
            priority=scenario.priority.value,
            steps_planned=len(scenario.steps_instructions),
            steps_instructions=[
                self._scrub(value) for value in scenario.steps_instructions
            ],
            expected_outcomes=[
                self._scrub(outcome.statement) for outcome in scenario.expected_outcomes
            ],
            outcome_uuids=[
                scenario.outcome_uuid(index)
                for index, _ in enumerate(scenario.expected_outcomes)
            ],
            required_metrics=[metric.value for metric in scenario.required_metrics],
        )

    def _scrub(self, value: Optional[str]) -> str:
        return self._scrubber.scrub(value) if self._scrubber else value or ""

    def start(self, test_id: str) -> None:
        entry = self.entries.get(test_id)
        if entry is not None:
            entry.status = ReportStatus.RUNNING
            entry.started_at = _now()
        self.write()

    def add(self, report: ScenarioReport) -> None:
        """Takes a row for something that was never a listed scenario.

        A cell nothing could cover, a report that failed to write, and usage
        that failed to serialize all reach the suite's reports. They have to
        reach this one too, or the record a reader is told to trust is the
        one artifact missing them.
        """
        self.entries[report.test_id] = ScenarioEntry(
            position=len(self.entries) + 1,
            scenario_uuid=report.scenario_uuid or new_uuid(),
            test_id=report.test_id,
            name=self._scrub(report.scenario_name),
            cell=report.cell,
            category="",
            scenario_type="",
            priority="",
            steps_planned=0,
            steps_instructions=[],
            expected_outcomes=[],
        )
        self.record(report.test_id, report)

    def record_or_add(self, report: ScenarioReport) -> None:
        """Record a planned scenario, or add an infrastructure row."""
        if report.test_id in self.entries:
            self.record(report.test_id, report)
        else:
            self.add(report)

    def record(
        self,
        test_id: str,
        report: ScenarioReport,
        calls: Optional[List[CallRow]] = None,
    ) -> None:
        """Takes everything one finished scenario learned."""
        entry = self.entries.get(test_id)
        if entry is not None:
            entry.calls = list(calls or [])
            entry.status = report.status
            entry.cause = report.cause
            entry.steps_executed = report.steps_executed
            entry.duration_seconds = round(report.duration, 1)
            entry.failure_reason = self._scrub(report.failure_reason)
            entry.issues_found = len(report.detected_issues or [])
            entry.steps = list(report.steps)
            entry.rule_verifications = list(report.rule_verifications)
            entry.detected_issues = list(report.detected_issues)
            entry.lifecycle_operations = [
                self._masked_lifecycle_operation(operation)
                for operation in report.lifecycle_operations
            ]
            entry.measurements = [self._masked_measurement(item) for item in report.measurements]
            entry.verdict_inputs = self._masked(report.verdict_inputs)
            entry.finished_at = _now()
        self.write()

    def _masked(self, inputs: Optional[VerdictInputs]) -> Optional[VerdictInputs]:
        """The judge's inputs, with every declared value masked.

        A finding quotes what was on screen, and a rule is written by whoever
        wrote the journey. Both reach this file, so both go through the
        scrubber like everything else in it.
        """
        if inputs is None:
            return None
        kept = inputs.model_copy(deep=True)
        kept.stop_reason = self._scrub(kept.stop_reason)
        kept.infeasibility_reason = self._scrub(kept.infeasibility_reason)
        kept.security_rules = [self._scrub(rule) or "" for rule in kept.security_rules]
        for outcome in kept.expected_outcomes:
            outcome.statement = self._scrub(outcome.statement) or ""
        for finding in kept.sensitive_findings:
            finding.quoted = self._scrub(finding.quoted) or ""
            finding.field = self._scrub(finding.field) or ""
        return kept

    def _masked_measurement(self, measurement: Measurement) -> Measurement:
        """A numeric sample with its optional target labels scrubbed."""
        return measurement.model_copy(
            update={
                "collector": self._scrub(measurement.collector),
                "collector_version": (
                    self._scrub(measurement.collector_version)
                    if measurement.collector_version
                    else None
                ),
                "os_version": (
                    self._scrub(measurement.os_version)
                    if measurement.os_version
                    else None
                ),
                "device_model": (
                    self._scrub(measurement.device_model)
                    if measurement.device_model
                    else None
                ),
                "screen": self._scrub(measurement.screen) if measurement.screen else None,
            }
        )

    def _masked_lifecycle_operation(
        self, operation: LifecycleOperation
    ) -> LifecycleOperation:
        """A lifecycle record with every device-supplied string scrubbed."""
        return operation.model_copy(
            update={
                "requested": self._scrub(operation.requested),
                "source": self._scrub(operation.source),
                "observed": (
                    self._scrub(operation.observed) if operation.observed else None
                ),
                "failure_reason": (
                    self._scrub(operation.failure_reason)
                    if operation.failure_reason
                    else None
                ),
            }
        )

    def record_unattached_calls(self, calls: List[CallRow]) -> None:
        """Takes the calls nothing in the run owns, such as generation."""
        self.calls = list(calls)

    def _elapsed(self) -> float:
        """How long the run took, by this file's own clock."""
        return round((datetime.now() - self.started_at).total_seconds(), 2)

    def finish(
        self,
        status: JourneyStatus,
        unevidenced: Optional[List[str]] = None,
        policy_checks: Optional[Sequence[PolicyCheckReport]] = None,
        usage: Optional[dict] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Closes the record, so a reader can tell a finished run from a killed one.

        An aborted run keeps that status. The suite still aggregates its
        scenarios afterwards and arrives at FAILED. That is true of the
        scenarios and not of the run. The run reached no verdict. It was
        interrupted, and the distinction is why the status exists.
        """
        if self.status != ABORTED:
            self.status = status
        self.unevidenced_security_rules = list(unevidenced or [])
        self.policy_checks = [self._masked_policy_check(check) for check in policy_checks or ()]
        if self.status != ABORTED and not policy_checks_succeeded(self.policy_checks):
            self.status = ReportStatus.FAILED
        self.usage = usage
        # The suite already timed itself. Its number is the one the reports
        # show, so the record states that rather than a second measurement.
        self.duration = duration if duration is not None else self._elapsed()
        self.write(final=True)

    def _masked_policy_check(self, check: PolicyCheckReport) -> PolicyCheckReport:
        """A run-level check with every free-text evidence field scrubbed."""
        return check.model_copy(
            update={
                "source": self._scrub(check.source),
                "evidence": self._scrub(check.evidence),
                "waiver_reason": (
                    self._scrub(check.waiver_reason) if check.waiver_reason else None
                ),
            }
        )

    def record_goal(self, result, usage: Optional[dict] = None) -> None:
        """Closes the record of a single-goal run.

        A goal run produces no scenario and no verdict. What it leaves is the
        walk: whether the goal was reached, and the steps it took trying.
        """
        self.steps = list(result.steps)
        self.usage = usage
        self.duration = self._elapsed()
        if self.status != ABORTED:
            self.status = (
                ReportStatus.PASSED
                if result.status == "goal_reached"
                else ReportStatus.FAILED
            )
        self.write(final=True)

    def abort(self, reason: str) -> None:
        """Marks everything unfinished as aborted, so a crash leaves no 'running' row."""
        for entry in self.entries.values():
            if entry.status.in_progress:
                entry.status = ReportStatus.ABORTED
                entry.failure_reason = self._scrub(reason)
                entry.finished_at = _now()
        self.status = ABORTED
        self.write(final=True)

    def _first_failure(self, entries: List[ScenarioEntry]) -> Optional[FailureSummary]:
        for entry in entries:
            if entry.status not in (ReportStatus.FAILED, ReportStatus.ERROR):
                continue
            failed = next(
                (
                    verify.rule_id
                    for verify in entry.rule_verifications
                    if verify.status == "failed"
                ),
                None,
            )
            return FailureSummary(
                scenario_uuid=entry.scenario_uuid,
                test_id=entry.test_id,
                rule_id=failed,
                step=entry.steps_executed,
            )
        return None

    def payload(self, final: bool = False) -> dict:
        """The record as it stands, which is what gets written."""
        entries = sorted(self.entries.values(), key=lambda entry: entry.position)
        counts: dict[ReportStatus, int] = {}
        for entry in entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1

        record = {
            "schema": SCHEMA,
            "run_uuid": self.run_uuid,
            "mode": self.mode,
            "status": self.status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_seconds": self.duration,
            "updated_at": _now(),
            "ended_at": _now() if final else None,
            "subject": self.subject.model_dump(),
            "identity": self.environment.model_dump(),
            "inputs": self.inputs.model_dump(),
            "total": len(entries),
            "counts": counts,
            "scenarios": [entry.model_dump(exclude_none=True) for entry in entries],
            "steps": [step.model_dump(exclude_none=True) for step in self.steps],
            "uncovered": [
                {"test_id": entry.test_id, "cell": entry.cell,
                 "status": entry.status, "cause": entry.cause,
                 "reason": entry.failure_reason}
                for entry in entries
                if _is_uncovered(entry)
            ],
            "calls": [row.model_dump(exclude_none=True) for row in self.calls],
            "unevidenced_security_rules": list(self.unevidenced_security_rules),
            "artifacts": list(self.artifacts),
            "policy_checks": [
                check.model_dump(mode="json", exclude_none=True)
                for check in self.policy_checks
            ],
            "usage": self.usage,
        }
        first = self._first_failure(entries)
        record["first_failure"] = first.model_dump() if first else None
        return record

    def write(self, final: bool = False) -> None:
        """Replaces the file whole, so a reader never sees half a record.

        A failure is remembered rather than raised. This is called after
        every scenario, and a run that reached verdicts must not be ended by
        the file that records them. The caller asks `unwritten` at the end
        and fails the run there, because a run with no record is not a run
        anybody can read.
        """
        try:
            body = json.dumps(self.payload(final), indent=2, default=str)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(self.path)
            self.unwritten = None
        except OSError as exc:
            logger.error("Could not update the run manifest at %s: %s", self.path, exc)
            self.unwritten = exc
