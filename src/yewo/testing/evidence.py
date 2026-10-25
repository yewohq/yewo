"""What a run leaves behind: the evidence, and the reports built from it.

Split from `models.py`, which holds what a run is given. These are what it
produces, and the rules a run answers, which sit beside the answers to them. One module answered both questions and grew past the file ceiling.

Every name here is re-exported by `models.py`, so no caller has to know
which of the two it came from.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from yewo.agent.actuation.actions import CapabilityKind
from yewo.agent.run.outcomes import RunStatus
from yewo.agent.run.postconditions import PostconditionVerification
from yewo.agent.run.record import StepRecord
from yewo.identity import AppIdentity, CiIdentity, DeviceIdentity, GitIdentity


# The model writes this one to point at what it reports. The detector's
# perception.BoundingBox stays separate, so neither schema moves when the other
# changes.
class BoundingBox(BaseModel):
    """Represents coordinates highlighting a visual issue."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Rendering(str, Enum):
    """How a sensitive value appears on screen.

    An exposure claim is only as good as what was seen. Asked for a sentence,
    a model can call a bare caption a legible value and pass the rule
    forbidding one in the same answer. Asked which of these it saw, it has to
    pick, and the four are not confusable.
    """

    VALUE = "value"
    """The real characters are legible. This is the exposure."""

    MASKED = "masked"
    """Dots, asterisks or bullets stand in for the characters."""

    LABEL_ONLY = "label_only"
    """A caption naming the field, with no value beside it."""

    ABSENT = "absent"
    """Nothing of the kind is on this screen."""


class SensitiveFinding(BaseModel):
    """What was seen where a sensitive value could be, and what it looked like."""

    field: str = Field(
        ..., description="What the value is, named as this screen names it."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_finding_fields(cls, value):
        """Reads a finding under the names its two fields arrive with.

        How a value was rendered comes back as `rendering`, `state`, `status`
        or `type`. `type` is the ambiguous one: it names the value in some
        answers and its rendering in others. The renderings are a closed set,
        so which one it carries can be read off rather than guessed.

        A missing name falls back to where the value was seen, which is what
        the model gave instead of naming it.
        """
        if not isinstance(value, dict):
            return value
        finding = dict(value)
        named = finding.pop("type", None)
        if named in {rendering.value for rendering in Rendering}:
            finding.setdefault("rendering", named)
        elif named is not None:
            finding.setdefault("field", named)
        for alias in ("state", "status"):
            if "rendering" not in finding and alias in finding:
                finding["rendering"] = finding.pop(alias)
        if "field" not in finding and "location" in finding:
            finding["field"] = finding["location"]
        if finding.get("quoted") is None:
            finding["quoted"] = ""
        return finding
    rendering: Rendering = Field(
        ...,
        description=(
            "Exactly what is on screen. 'value' only when the real characters "
            "are legible. A caption with nothing beside it is 'label_only'. "
            "Dots or bullets are 'masked'."
        ),
    )
    quoted: str = Field(
        default="",
        description="The characters on screen, copied exactly. Empty unless rendering is 'value'.",
    )
    screen_name: Optional[str] = Field(
        default=None,
        description="Filled in by Yewo, not by the model, from the screen audited.",
    )

    def is_exposure(self) -> bool:
        """Whether this is a leak, decided by the rendering rather than by prose."""
        return self.rendering is Rendering.VALUE


class ReportStatus(str, Enum):
    """Every status persisted in a run record or shown in a report."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    ABORTED = "ABORTED"
    UNVERIFIED = "UNVERIFIED"
    UNSUPPORTED = "UNSUPPORTED"

    def __str__(self) -> str:
        return self.value

    @property
    def in_progress(self) -> bool:
        """Whether the scenario has not produced a result yet."""
        return self in (ReportStatus.PENDING, ReportStatus.RUNNING)


ScenarioStatus = Literal[
    ReportStatus.PASSED,
    ReportStatus.FAILED,
    ReportStatus.SKIPPED,
    ReportStatus.ERROR,
    ReportStatus.ABORTED,
    ReportStatus.UNVERIFIED,
    ReportStatus.UNSUPPORTED,
]
"""A status a finished scenario can report."""

JourneyStatus = Literal[
    ReportStatus.PASSED,
    ReportStatus.FAILED,
    ReportStatus.RUNNING,
]
"""A status an aggregated journey report can carry."""

REPORT_STATUS_ALIASES = {
    "pass": ReportStatus.PASSED,
    "passed": ReportStatus.PASSED,
    "fail": ReportStatus.FAILED,
    "failed": ReportStatus.FAILED,
}
"""Accepted non-canonical spellings in a stored record."""


class IssueKind(str, Enum):
    """What kind of thing a finding is.

    Closed, because a reader is shown one row per defect. Rows gather only
    by a value that means the same thing twice. Asked for free text, one
    auditor gave a single debug ribbon four names across four screens.
    Nothing could tell they were one defect.
    """

    LAYOUT_OVERFLOW = "layout_overflow"
    TEXT_CLIPPING = "text_clipping"
    OVERLAPPING_ELEMENTS = "overlapping_elements"
    DEBUG_ARTIFACT = "debug_artifact"
    CRASH_SCREEN = "crash_screen"
    EMPTY_STATE_ERROR = "empty_state_error"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    UNMASKED_INPUT = "unmasked_input"
    INSECURE_CONNECTION = "insecure_connection"
    BROKEN_NAVIGATION = "broken_navigation"
    ACCESSIBILITY_VIOLATION = "accessibility_violation"
    PERFORMANCE_HANG = "performance_hang"
    OTHER = "other"


class QAIssue(BaseModel):
    """Represents a visual, security, or logical issue found during testing."""
    issue_type: IssueKind = Field(
        ...,
        description=(
            "Which kind of issue this is. Pick the closest one. Use 'other' "
            "only where none of them describes what you saw."
        ),
    )

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalize_issue_kind(cls, value):
        """Reads a kind however it was cased, or files it as other.

        Case, spaces and hyphens only. Synonyms are not mapped here. Which
        words one model reaches for is a fact about that model and that
        app. A list of them would be a list of one app's accidents. The
        set is closed in the schema, so a model is asked to pick from it
        rather than to describe.

        A kind that still does not match is `other` rather than refused. A
        finding is evidence, and dropping one because its label was wrong
        loses the defect with the word. `other` filling up is the signal
        that the set or the prompt needs work.
        """
        if isinstance(value, IssueKind):
            return value
        if not isinstance(value, str):
            return IssueKind.OTHER
        normalized_kind = value.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return IssueKind(normalized_kind)
        except ValueError:
            return IssueKind.OTHER
    description: str
    bounding_box: Optional[BoundingBox] = None
    severity: Literal["low", "medium", "high", "critical"]
    current_screen: Optional[str] = None
    step: Optional[int] = Field(
        default=None,
        description=(
            "Which step the screen was audited after. Stamped by the run "
            "and not answered: an auditor is shown one screen and does not "
            "know where in a walk it sits. It is what lets a report link a "
            "finding to the screenshot that shows it."
        ),
    )


STATUS_SPELLINGS = {
    "pass": "passed",
    "passed": "passed",
    "fail": "failed",
    "failed": "failed",
    "untested": "untested",
    "not_tested": "untested",
}
"""How a verdict's status is spelled, mapped to the three that are accepted."""


class AssertionScope(str, Enum):
    """What kind of evidence an expected outcome needs.

    The auditor is handed one screenshot. Asked whether an app hung, or
    whether it navigated somewhere, it answers from pixels that cannot carry
    either fact. Both a false pass and a false missing-evidence follow. The
    scope says which evidence settles the claim, so a claim nothing can settle
    is visible instead of silently wrong.
    """

    CURRENT_SCREEN = "current_screen"
    """True or false about what is on one screen. The auditor can see this."""

    TRANSITION = "transition"
    """The run reached a screen, or moved between two. The screens visited
    carry this. A single screenshot does not."""

    RUN_TIMING = "run_timing"
    """How long the app took, or whether it stayed responsive. Nothing Yewo
    records measures this today, so an outcome scoped here is reported as
    unproven rather than counted either way."""


class ExpectedOutcome(BaseModel):
    """One thing a scenario must show, and what could show it."""

    statement: str
    scope: AssertionScope = Field(
        default=AssertionScope.CURRENT_SCREEN,
        description=(
            "Which evidence settles this. 'current_screen' for anything visible "
            "on a single screen. 'transition' for having reached or moved "
            "between screens. 'run_timing' for speed or responsiveness."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_expected_outcome(cls, value):
        """A plain string is a claim about one screen, which is the common case."""
        if isinstance(value, str):
            return {"statement": value}
        return value


class RuleVerification(BaseModel):
    """Represents the status of verifying a specific journey expectation or custom rule."""
    rule: str = Field(
        ...,
        description="The words of the rule being answered, as they were listed.",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable ID supplied in the prompt, such as expected:0 or security:1, "
            "and nothing else. Dynamic assertions may omit it."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_rule_verification(cls, value):
        """Reads a verdict however the model chose to write it.

        One rejected verdict takes the whole assessment with it, and the
        verdicts beside it were answers to other rules.

        The gate needs the id and matches it whole. The id is read from any
        field that holds it and from surrounding text. It stands in for `rule`
        when only the id was given. An id names its rule as well as the words
        do.

        An unrecognised status is left alone to be refused. A verdict is the
        thing being recorded, and guessing at one invents evidence.
        """
        if not isinstance(value, dict):
            return value
        from yewo.testing.judge import rules

        verdict = dict(value)
        found = rules.id_in(verdict.get("rule_id")) or rules.id_in(verdict.get("rule"))
        if found:
            verdict["rule_id"] = found
        if not verdict.get("rule") and verdict.get("rule_id"):
            verdict["rule"] = verdict["rule_id"]
        verdict["status"] = STATUS_SPELLINGS.get(
            str(verdict.get("status", "")).strip().lower().replace(" ", "_"),
            verdict.get("status"),
        )
        return verdict
    status: Literal["passed", "failed", "untested"]
    applicable: bool = Field(
        default=True,
        description=(
            "False when this screen has nothing the rule could apply to, so a "
            "'passed' here would mean nothing. A rule about a field is not "
            "applicable on a screen without that field."
        ),
    )
    reasoning: str
    screen_name: Optional[str] = None


class ReceiptRecord(BaseModel):
    """What the device reported about one capability a scenario declared.

    Two fields, because two are what the verdict reads: which capability, and
    whether the device reported the state. The states themselves are left
    out. A clipboard receipt carries whatever was copied, which can be a
    password. The verdict never looks at the value.
    """

    capability: str
    confirmed: bool


class VerdictInputs(BaseModel):
    """Everything a scenario's verdict was decided from.

    The record kept the verdict and the auditor's answers, and not the rest
    of what the judge was given. So the record could not reach its own
    verdict again. A rule change could not be measured against a run that
    already happened.

    Held together rather than spread across the report, because they are one
    thing: the question the judge was asked. A reader that wants to ask it
    again needs all of them or none.

    Each field is one the judge reads, and nothing here is kept for any
    other reader.
    """

    run_status: RunStatus
    stop_reason: Optional[str] = None
    is_scenario_feasible: bool = Field(
        default=True,
        description=(
            "What the planner last answered about proceeding. Named as the "
            "planner names it, because this stands in for that answer when a "
            "stored run is decided again."
        ),
    )
    infeasibility_reason: Optional[str] = None
    screens_judged: int = Field(
        default=0, description="How many screens the auditor answered about."
    )
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list)
    security_rules: List[str] = Field(default_factory=list)
    sensitive_findings: List[SensitiveFinding] = Field(default_factory=list)
    required_capabilities: List[CapabilityKind] = Field(default_factory=list)
    receipts: List[ReceiptRecord] = Field(default_factory=list)
    postcondition_verifications: List[PostconditionVerification] = Field(
        default_factory=list
    )


class ScenarioReport(BaseModel):
    """Represents the results of running an individual test scenario."""
    scenario_uuid: Optional[str] = Field(
        default=None,
        description=(
            "Which scenario this reports on. Absent on a report Yewo writes "
            "about itself, which answered no scenario."
        ),
    )
    test_id: str
    scenario_name: str
    cell: str = Field(
        description=(
            "The cell this scenario answered, such as 'security.negative.0'. "
            "A report Yewo writes about itself rather than a scenario carries "
            "'infrastructure'."
        )
    )
    status: ScenarioStatus
    cause: Optional[str] = Field(
        default=None,
        description=(
            "Whose problem a scenario that reached no verdict is: the device, "
            "a model provider, Yewo, the evidence or the scenario itself. "
            "Absent on PASSED and on FAILED, where the app is the answer."
        ),
    )
    duration: float = 0.0
    steps_executed: int = 0
    rule_verifications: List[RuleVerification] = Field(default_factory=list)
    detected_issues: List[QAIssue] = Field(default_factory=list)
    screenshots: List[str] = Field(default_factory=list)
    steps: List[StepRecord] = Field(
        default_factory=list,
        description="What each step of this scenario did, in order.",
    )
    postcondition_verifications: List[PostconditionVerification] = Field(
        default_factory=list,
        description="Secret-free local verifications that ran before dependent actions.",
    )
    failure_reason: Optional[str] = None
    verdict_inputs: Optional[VerdictInputs] = Field(
        default=None,
        description=(
            "What the verdict was decided from, so it can be decided again. "
            "Absent on a report Yewo writes about itself, which no judge saw."
        ),
    )


class RunEnvironment(BaseModel):
    """What produced a report, for when the same journey behaves differently.

    Yewo's own settings, and the four things around them that decide what a
    run means: the build, the target, the commit and the job. A scenario that
    failed, with no record of which build it failed on, is not evidence of a
    defect.
    """
    yewo_version: str
    model: str
    fast_model: Optional[str] = None
    platform: str
    grounding: str
    detector: Optional[str] = Field(
        default=None, description="Detection weights, as repository@revision."
    )
    app: AppIdentity = Field(default_factory=AppIdentity)
    device: DeviceIdentity = Field(default_factory=DeviceIdentity)
    git: GitIdentity = Field(default_factory=GitIdentity)
    ci: CiIdentity = Field(default_factory=CiIdentity)


class JourneyReport(BaseModel):
    """Represents the aggregated results of all scenarios generated for a journey."""
    journey_id: str
    journey_name: str
    description: str
    status: JourneyStatus
    total_duration: float = 0.0
    scenarios: List[ScenarioReport] = Field(default_factory=list)
    unevidenced_security_rules: List[str] = Field(
        default_factory=list,
        description=(
            "Security rules no scenario was able to evidence. A rule no "
            "scenario reached is not a rule that held."
        ),
    )
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    environment: Optional[RunEnvironment] = None
