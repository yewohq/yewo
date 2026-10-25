"""What the page reads from a run, whoever supplies it.

Two suppliers, and the page cannot tell them apart. A run that is finishing
hands over the reports it holds in memory. A reader opening `run.json`
afterwards hands over the record. Both answer to the names below, so what the
page says about a run does not depend on which one opened it.

Read-only on purpose. The page decides what a reader is shown. It never
decides what a run recorded, and it writes nothing back.
"""

from typing import Optional, Protocol, Sequence

from yewo.agent.run.record import StepRecord
from yewo.testing.models import (
    PolicyCheckReport,
    QAIssue,
    ReportStatus,
    RuleVerification,
    RunEnvironment,
    VerdictInputs,
)
from yewo.testing.lifecycle import LifecycleOperation


class ScenarioSource(Protocol):
    """One scenario, under the names the page asks for it by.

    A record is a file and can contain any status text. Its reader validates
    that text before the page receives this enum.
    """

    @property
    def test_id(self) -> str: ...

    @property
    def scenario_name(self) -> str: ...

    @property
    def cell(self) -> str: ...

    @property
    def status(self) -> ReportStatus: ...

    @property
    def cause(self) -> Optional[str]: ...

    @property
    def duration(self) -> float: ...

    @property
    def steps_executed(self) -> int: ...

    @property
    def failure_reason(self) -> Optional[str]: ...

    @property
    def steps(self) -> Sequence[StepRecord]: ...

    @property
    def rule_verifications(self) -> Sequence[RuleVerification]: ...

    @property
    def detected_issues(self) -> Sequence[QAIssue]: ...

    @property
    def lifecycle_operations(self) -> Sequence[LifecycleOperation]: ...

    @property
    def verdict_inputs(self) -> Optional[VerdictInputs]:
        """What the judge was given, which holds the rules as written.

        A verification carries the rule as the auditor echoed it back. An
        auditor echoes differently each time. It gives the full line, the
        line with its id, or the id alone. The journey's own words are
        here.
        """
        ...


class RunSource(Protocol):
    """One run, under the names the page asks for it by."""

    @property
    def journey_id(self) -> str: ...

    @property
    def journey_name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def status(self) -> ReportStatus: ...

    @property
    def total_duration(self) -> float: ...

    @property
    def passed_count(self) -> int: ...

    @property
    def skipped_count(self) -> int: ...

    @property
    def scenarios(self) -> Sequence[ScenarioSource]: ...

    @property
    def policy_checks(self) -> Sequence[PolicyCheckReport]: ...

    @property
    def unevidenced_security_rules(self) -> Sequence[str]: ...

    @property
    def environment(self) -> Optional[RunEnvironment]: ...
