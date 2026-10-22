"""One scenario's walk, as the rows a reader opens.

Separate from the page's account of the run. A step is read by somebody
who already knows the verdict and wants to see how it was reached. What a
step shows is a different question from what a report concludes.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from yewo.agent.run.record import StepRecord

from .facts import seconds
from .source import ScenarioSource

def anchor_of(value: str) -> str:
    """An id a browser can jump to, from an identifier an app supplied."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return cleaned.lower() or "scenario"


CAPTIONS = {
    "raw": "Screen Before Action",
    "annotated": "Perception Overlay",
    "action": "Action Target",
    "scan": "Element Hierarchy",
}
"""What each file a step wrote shows, named for a reader rather than a path.

`annotated` is what the detector found. `action` is where it landed. The
first one was captioned as the thing Yewo aimed at, which named the wrong
image.
"""

IMAGES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class Chip:
    """A short label beside a step or a status."""

    label: str
    tone: str = "muted"


@dataclass(frozen=True)
class Note:
    """One labelled paragraph of what a step recorded."""

    label: str
    body: str


@dataclass(frozen=True)
class Shot:
    """A file a step wrote, and how it is shown."""

    caption: str
    href: str
    image: bool


@dataclass(frozen=True)
class StepView:
    """One row of a scenario's walk."""

    number: str
    tone: str
    title: str
    duration: str
    anchor: str
    chips: Tuple[Chip, ...] = ()
    notes: Tuple[Note, ...] = ()
    shots: Tuple[Shot, ...] = ()
    opened: bool = False

    @property
    def expandable(self) -> bool:
        return bool(self.notes or self.shots)


def _took(step: StepRecord) -> str:
    """How long one step took, over the three phases that are timed."""
    spent = [step.perceive_ms, step.plan_ms, step.act_ms]
    measured = [value for value in spent if value is not None]
    return seconds(sum(measured) / 1000) if measured else ""


def _notes(step: StepRecord) -> Tuple[Note, ...]:
    """What the step recorded, under the names a reader asks the questions in."""
    decision = step.decision
    written = [
        ("Screen", step.screen_name),
        ("Observed State", decision.observation if decision else ""),
        ("Action Reasoning", decision.reasoning if decision else ""),
        ("Previous Step Evaluation", decision.evaluation_previous_goal if decision else ""),
        ("Goal Evidence", decision.goal_evidence if decision else ""),
        ("Action", step.action.described if step.action else ""),
        ("Target Element", step.action.target_label if step.action else ""),
        ("Action Refused", step.grounding.refusal if step.grounding else ""),
        ("Device Response", step.outcome.observation if step.outcome else ""),
        ("Error", step.outcome.error if step.outcome else ""),
    ]
    return tuple(Note(label, body) for label, body in written if body)


def _shots(step: StepRecord, href: str) -> Tuple[Shot, ...]:
    """The files the step wrote, linked where this page can reach them."""
    order = {name: index for index, name in enumerate(CAPTIONS)}
    named = sorted(
        step.screenshots.items(), key=lambda pair: (order.get(pair[0], 99), pair[0])
    )
    return tuple(
        Shot(
            caption=CAPTIONS.get(kind, kind.replace("_", " ").capitalize()),
            href=f"{href}/{filename}" if href else filename,
            image=filename.lower().endswith(IMAGES),
        )
        for kind, filename in named
    )


def _chips(step: StepRecord, recovered: bool = False) -> Tuple[Chip, ...]:
    """The labels that say how this step went, beside the title.

    A failed action inside a scenario that went on to pass is said to
    have recovered. The bare word next to a green verdict reads as the
    scenario's own result.
    """
    chips: List[Chip] = []
    if step.source == "replay":
        chips.append(Chip("Replayed", "muted"))
    if step.grounding is not None and not step.grounding.resolved:
        label = "Target not on screen" if step.grounding.unmatched else "Not sent"
        chips.append(Chip(label, "warn"))
    if step.outcome is not None and not step.outcome.success:
        if recovered:
            chips.append(Chip("Action failed · recovered", "warn"))
        else:
            chips.append(Chip("Action failed", "fail"))
    return tuple(chips)


def _title(step: StepRecord) -> str:
    """What the step was for, in the words of whoever decided it.

    The intent rather than the mechanics. A row reading `tap at (540,
    1644)` makes a reader work out what was tapped. The coordinates are
    in the detail below it either way.
    """
    action = step.action
    if action and action.intent:
        return action.intent
    if action and action.described:
        return action.described
    if step.decision and step.decision.reasoning:
        return step.decision.reasoning
    return f"Step {step.n}"


def failed_at(steps: Sequence[StepRecord]) -> Optional[int]:
    """The first step the device did not do, which is where a reader looks."""
    for step in steps:
        if step.outcome is not None and not step.outcome.success:
            return step.n
    return None


def step_views(
    scenario: ScenarioSource, href: str, opened: Optional[int]
) -> Tuple[StepView, ...]:
    recovered = scenario.status == "PASSED"
    views = []
    for step in scenario.steps:
        tone = "muted"
        if step.outcome is not None:
            tone = "pass" if step.outcome.success else "warn" if recovered else "fail"
        views.append(
            StepView(
                number=f"{step.n:02d}",
                tone=tone,
                title=_title(step),
                duration=_took(step),
                anchor=f"{anchor_of(scenario.test_id)}-step-{step.n}",
                chips=_chips(step, recovered),
                notes=_notes(step),
                shots=_shots(step, href),
                opened=step.n == opened,
            )
        )
    return tuple(views)
