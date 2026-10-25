"""Reading the eval corpus, and turning a fixture into real objects.

A fixture is data, not Python. That is the whole point of it. A unit test and
the code it tests move together, because whoever changes the code can change
the test in the same edit and the suite stays green. A fixture file holds the
answer a maintainer decided on, in a file no refactor has a reason to touch.

Two kinds live here.

- A **verdict fixture** holds the evidence a run produced and the one word it
  must add up to. It needs no model and no device, so it is exact.
- A **screen fixture** holds a screenshot, the rules to put to it, and the
  answer each rule must get. It needs a model, so it is read over several
  replicates and reported as stable or not.

Every file is read through a model that refuses an unknown field and a wrong
shape. A fixture is the thing a gate trusts, so nothing about one is taken on
the strength of a key being present.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from yewo.agent.actuation.actions import CapabilityKind
from yewo.agent.device.receipts import Receipt
from yewo.agent.memory.history import AgentHistory
from yewo.agent.run.outcomes import DeviceRunResult, RunStatus
from yewo.errors import Cause
from yewo.testing.models import (
    ExpectedOutcome,
    Measurement,
    QAIssue,
    RuleVerification,
    ScenarioStatus,
    SensitiveFinding,
)

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "eval"
VERDICTS = CORPUS_ROOT / "verdicts"
SCREENS = CORPUS_ROOT / "screens"

VERDICT_TIER = "verdicts"
SCREEN_TIER = "screens"

PNG = b"\x89PNG\r\n\x1a\n"
"""What a PNG starts with. A fixture names a screenshot, and the bytes are
sent to a model, so the file is read as a picture before it is sent as one."""


class CorpusError(Exception):
    """A fixture that cannot be read. Never treated as an empty corpus."""


DIGEST_LENGTH = 16
"""Characters of a fixture's hash a baseline entry carries."""


def _digest(*parts: bytes) -> str:
    """What a fixture says, as a hash a baseline entry can be bound to.

    Over the bytes of the file, and of the screenshot where there is one. An
    exemption is granted to a case, not to a name. Edit the evidence or the
    label under a name an exemption already covers and the exemption has to
    lapse, or a corpus can be weakened without the gate saying so.
    """
    running = hashlib.sha256()
    for part in parts:
        running.update(part)
        running.update(b"\0")
    return running.hexdigest()[:DIGEST_LENGTH]


def _plain_name(value: str) -> str:
    """One fixture id, refusing the separator that divides a screen's items.

    A screen rule is reported as `fixture/expected:0`. A verdict fixture
    naming itself with a slash could take that name, and a baseline entry
    would then cover whichever of the two it reached first.
    """
    if "/" in value:
        raise ValueError("an id holds no '/'")
    return value


class _Receipt(BaseModel):
    """What the device reported about one capability, as a fixture says it."""

    model_config = ConfigDict(extra="forbid")

    capability: str
    requested: str = "asked"
    confirmed: bool = True
    target: str = "the fixture's target"
    source: str = "eval corpus"

    def built(self) -> Receipt:
        return Receipt(
            capability=self.capability,
            requested=self.requested,
            observed=self.requested if self.confirmed else None,
            target=self.target,
            source=self.source,
        )


class _PlannerAnswer(BaseModel):
    """The two fields a verdict reads off what the planner last answered.

    The planner's own response model carries the whole next action. None of
    that decides a verdict, and a fixture that filled it in would be asserting
    on a schema this corpus is not about.
    """

    model_config = ConfigDict(extra="forbid")

    feasible: bool = True
    infeasibility_reason: Optional[str] = None

    @property
    def is_scenario_feasible(self) -> bool:
        return self.feasible


class _Run(BaseModel):
    """How the walk ended, and what the device confirmed along the way."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    steps_executed: int = 1
    failure_reason: Optional[str] = None
    cause: Optional[Cause] = None
    receipts: list[_Receipt] = Field(default_factory=list)

    def built(self) -> DeviceRunResult:
        return DeviceRunResult(
            status=self.status,
            steps_executed=self.steps_executed,
            failure_reason=self.failure_reason,
            history=AgentHistory(),
            capability_receipts=[given.built() for given in self.receipts],
            cause=self.cause,
        )


class _Expect(BaseModel):
    """What a verdict fixture says the judge must answer."""

    model_config = ConfigDict(extra="forbid")

    status: ScenarioStatus
    cause: Optional[Cause] = None
    reason_holds: Optional[str] = None


class _VerdictFile(BaseModel):
    """One verdict fixture, as it is written down."""

    model_config = ConfigDict(extra="forbid")

    id: str
    why: str
    run: _Run
    expect: _Expect
    stop_reason: Optional[str] = None
    verifications: list[RuleVerification] = Field(default_factory=list)
    issues: list[QAIssue] = Field(default_factory=list)
    findings: list[SensitiveFinding] = Field(default_factory=list)
    planner: list[_PlannerAnswer] = Field(default_factory=list)
    screens_judged: int = 1
    outcomes: list[ExpectedOutcome] = Field(default_factory=list)
    security_rules: list[str] = Field(default_factory=list)
    required_capabilities: list[CapabilityKind] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)

    _no_slash = field_validator("id")(_plain_name)


class _Rule(BaseModel):
    """One rule put to a screen, and the answer it must get."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kind: Literal["expected", "security"]
    expect: Literal["passed", "failed", "untested"]
    inverted: bool = False


class _ScreenFile(BaseModel):
    """One screen fixture, as it is written down."""

    model_config = ConfigDict(extra="forbid")

    id: str
    why: str
    screenshot: str
    rules: list[_Rule] = Field(min_length=1)

    _no_slash = field_validator("id")(_plain_name)

    @field_validator("rules")
    @classmethod
    def _one_rule_inverts_another(cls, rules: list) -> list:
        """A corpus of rules that are all true scores a reader that agrees."""
        if not any(rule.inverted for rule in rules):
            raise ValueError(
                "no inverted rule. A screen fixture needs one rule whose correct "
                "answer is the opposite of its twin, or it cannot tell reading "
                "from agreeing"
            )
        return rules

    @field_validator("screenshot")
    @classmethod
    def _a_png_beside_the_fixture(cls, name: str) -> str:
        """A plain file name, in the corpus, ending in `.png`.

        The bytes are sent to a model. A fixture that could name any path on
        the machine would be a way to send a file off it.
        """
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"'{name}' is not a plain file name beside the fixture")
        if not name.endswith(".png"):
            raise ValueError(f"'{name}' is not a .png")
        return name


@dataclass(frozen=True)
class Expectation:
    """What the judge must answer, as the gate compares it.

    The status is held exactly. The reason is held as a fragment it must
    contain, because the sentence is prose and prose gets rewritten. A rule id
    or a named phrase is the part that carries the meaning.

    The cause is checked only where the fixture names the key. `"cause": null`
    means the verdict must carry none, which is a real claim: a crash and a
    failed rule are the app's doing, and a cause would blame the run.
    """

    status: str
    cause: Optional[str] = None
    checks_cause: bool = False
    reason_holds: Optional[str] = None

    def disagreement(self, status: str, reason: Optional[str], cause) -> Optional[str]:
        """How the answer differs from this, or None where it agrees."""
        if status != self.status:
            return f"status {status}, expected {self.status}"
        named = cause.value if isinstance(cause, Cause) else cause
        wanted = self.cause.value if isinstance(self.cause, Cause) else self.cause
        if self.checks_cause and named != wanted:
            return f"cause {named}, expected {wanted}"
        if self.reason_holds and self.reason_holds not in (reason or ""):
            return f"reason does not hold '{self.reason_holds}': {reason!r}"
        return None


@dataclass(frozen=True)
class VerdictFixture:
    """One run's evidence, and the verdict it must produce."""

    id: str
    why: str
    expect: Expectation
    arguments: dict = field(default_factory=dict)
    path: Optional[Path] = None
    digest: str = ""


@dataclass(frozen=True)
class RuleAsked:
    """One rule put to a screen, and the answer it must get."""

    text: str
    kind: str
    expect: str
    inverted: bool = False


@dataclass(frozen=True)
class ScreenFixture:
    """One screenshot, the rules put to it, and the answers they must get."""

    id: str
    why: str
    screenshot: Path
    rules: list[RuleAsked]
    path: Optional[Path] = None
    digest: str = ""

    @property
    def outcomes(self) -> list[str]:
        return [rule.text for rule in self.rules if rule.kind == "expected"]

    @property
    def security_rules(self) -> list[str]:
        return [rule.text for rule in self.rules if rule.kind == "security"]


def _files(root: Path, kind: str) -> list[Path]:
    """The fixture files, refusing a directory that holds none.

    An empty corpus answers every question correctly and reports `0 of 0`. A
    wrong path, a checkout that did not fetch it, or a packaging mistake would
    then turn the gate off and pass the build.
    """
    found = sorted(root.glob("*.json"))
    if not found:
        raise CorpusError(f"no {kind} fixtures in {root}. A corpus is never empty.")
    return found


def _read(path: Path, model):
    """One fixture file, through the model that decides its shape."""
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise CorpusError(f"{path.name}: {_first_line(exc)}") from exc


NAMED = 3
"""How many of a report's errors are shown. One fixture rarely has more."""


def _first_line(exc: Exception) -> str:
    """One failure, said in a line a person can act on.

    A validation report runs to several lines per error, and its first line
    is a header naming the model. So each error is rendered as the field it
    is about and what was wrong with it.
    """
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:NAMED]
        )
    return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def verdict_fixtures(root: Path = VERDICTS) -> list[VerdictFixture]:
    """Every verdict fixture, in a fixed order so two runs read alike."""
    return [_verdict_fixture(path) for path in _files(root, VERDICT_TIER)]


def _verdict_fixture(path: Path) -> VerdictFixture:
    read: _VerdictFile = _read(path, _VerdictFile)
    return VerdictFixture(
        id=read.id,
        why=read.why,
        expect=Expectation(
            status=read.expect.status,
            cause=read.expect.cause,
            checks_cause="cause" in read.expect.model_fields_set,
            reason_holds=read.expect.reason_holds,
        ),
        arguments={
            "run_result": read.run.built(),
            "stop_reason": read.stop_reason,
            "all_verifications": read.verifications,
            "all_issues": read.issues,
            "qa_outputs": read.planner,
            "assessments": list(range(read.screens_judged)),
            "required_outcomes": read.outcomes,
            "security_rules": read.security_rules,
            "all_findings": read.findings,
            "required_capabilities": read.required_capabilities,
            "measurement_samples": read.measurements,
        },
        path=path,
        digest=_digest(path.read_bytes()),
    )


def screen_fixtures(root: Path = SCREENS) -> list[ScreenFixture]:
    """Every screen fixture, each with its screenshot beside it."""
    return [_screen_fixture(path) for path in _files(root, SCREEN_TIER)]


def _screen_fixture(path: Path) -> ScreenFixture:
    read: _ScreenFile = _read(path, _ScreenFile)
    screenshot = _screenshot(path, read.screenshot)
    return ScreenFixture(
        id=read.id,
        why=read.why,
        screenshot=screenshot,
        rules=[
            RuleAsked(rule.text, rule.kind, rule.expect, rule.inverted)
            for rule in read.rules
        ],
        path=path,
        digest=_digest(path.read_bytes(), screenshot.read_bytes()),
    )


def _screenshot(fixture: Path, name: str) -> Path:
    """The picture this fixture is about, read as one before it is sent as one.

    The name is already a plain file name. This resolves it anyway and checks
    where it landed, because a link inside the corpus would still point out of
    it.
    """
    directory = fixture.parent.resolve()
    path = (directory / name).resolve()
    if path.parent != directory:
        raise CorpusError(f"{fixture.name}: '{name}' is outside {directory}")
    if not path.is_file():
        raise CorpusError(f"{fixture.name}: no screenshot at {name}")
    if (directory / name).is_symlink():
        raise CorpusError(f"{fixture.name}: '{name}' is a link, not a screenshot")
    with path.open("rb") as handle:
        if handle.read(len(PNG)) != PNG:
            raise CorpusError(f"{fixture.name}: '{name}' is not a PNG")
    return path
