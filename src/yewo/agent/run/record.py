"""One step of a run, written down as it happens.

A report says a scenario failed and how many steps it took. It cannot say
which step, what the agent believed at the time, what it aimed at, or what
the device did about it. Those facts were logged in prose and then thrown
away, so every question about a past run was answered by rereading a log.

Each record is built across one turn of the loop: the screen, then the plan,
then the target, then what the device reported. A step that ends the run
early keeps the fields it reached and leaves the rest absent. Absent means
the step never got that far, and is never written as a zero or an empty
string that would read as a result.

Every string here is persisted. So every string here is scrubbed, by
whoever writes it down.
"""

from enum import Enum
from typing import Callable, List, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from yewo.llm.usage import CallRecord
from yewo.identity import new_uuid


class DecisionRecord(BaseModel):
    """What the planner said, in its own words.

    All five of its prose fields, not the reasoning alone. The honest
    self-assessment of the previous step is the field most worth reading when
    a scenario failed. It is also the one a single-field record drops.
    """

    current_screen: str = ""
    evaluation_previous_goal: str = ""
    observation: str = ""
    reasoning: str = ""
    goal_satisfied: bool = False
    goal_evidence: str = ""


class PlannedAction(Protocol):
    """What every action the planner can return declares.

    A protocol rather than `getattr`. Each action model names its own
    `reason` as a required field, so reading one is a typed contract and
    not a guess about what arrived.
    """

    @property
    def type(self) -> object: ...

    @property
    def reason(self) -> str: ...


class ActionRecord(BaseModel):
    """What the step did to the device, after the target was resolved."""

    type: str
    description: str = Field(description="The action in words, already redacted.")
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Why the planner chose this action, in its own words. Kept "
            "apart from the description so a reader can be shown what a "
            "step was for without the coordinates it was done at."
        ),
    )
    target_label: Optional[str] = Field(
        default=None, description="What the element said, where one was matched."
    )

    @classmethod
    def of(
        cls,
        action: PlannedAction,
        description: str,
        target: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> "ActionRecord":
        """The record for one action, read off the action itself."""
        kind = action.type
        return cls(
            type=kind.value if isinstance(kind, Enum) else str(kind),
            description=description,
            intent=intent or None,
            target_label=target,
        )


class GroundingRecord(BaseModel):
    """How the planner's target became something on the screen."""

    mode: str
    resolved: bool
    mark_id: Optional[int] = Field(
        default=None,
        description=(
            "The badge the planner answered with, in marks mode. Read from "
            "the planner's own action: the action that reaches the device "
            "carries coordinates and no badge."
        ),
    )
    refusal: Optional[str] = Field(
        default=None, description="Why the action never reached the device."
    )
    unmatched: bool = Field(
        default=False,
        description="The planner named a target this screen does not list.",
    )


class OutcomeRecord(BaseModel):
    """What the device did about the action."""

    success: bool
    observation: str
    error: Optional[str] = None


class CallRow(BaseModel):
    """One model call, as the record of a step accounts for it.

    The accounting only. The response body stays in `llm_calls.jsonl`, which
    is appended as each call returns and survives a crash that takes the
    manifest with it. Neither file repeats the other.
    """

    call_uuid: str = Field(default_factory=new_uuid)
    component: str = Field(
        description="Which phase of the run made this call: the generator, or the agent."
    )
    answering: Optional[str] = Field(
        default=None,
        description=(
            "The shape the answer had to take, by name. This is what tells "
            "the planner from the auditor: they share a client, and they ask "
            "for different things."
        ),
    )
    requested_model: str
    served_model: str
    generation_id: str
    attempt: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost: Optional[float]
    latency_ms: Optional[float]
    prompt_sha256: Optional[str] = None
    prompt_ref: Optional[str] = Field(
        default=None, description="Where the rendered prompt was written, if it was."
    )
    schema_sha256: Optional[str] = None

    @classmethod
    def of(cls, component: str, call: CallRecord) -> "CallRow":
        """The row for one call, read off the record the tracker already built."""
        return cls(
            component=component,
            answering=call.answering,
            requested_model=call.model,
            served_model=call.served_model,
            generation_id=call.generation_id,
            attempt=call.attempt,
            finish_reason=call.finish_reason,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            cached_tokens=call.cached_tokens,
            reasoning_tokens=call.reasoning_tokens,
            cost=call.cost,
            latency_ms=call.latency_ms,
            prompt_sha256=call.prompt_sha256,
            prompt_ref=call.prompt_ref,
            schema_sha256=call.schema_sha256,
        )


class StepRecord(BaseModel):
    """One turn of the loop: what was seen, decided, aimed at and done."""

    step_uuid: str = Field(default_factory=new_uuid)
    n: int = Field(description="Which step of the scenario this was, from 1.")
    source: Literal["model", "replay"] = "model"
    screen_name: str = ""
    phash: Optional[str] = None

    decision: Optional[DecisionRecord] = None
    action: Optional[ActionRecord] = None
    grounding: Optional[GroundingRecord] = None
    outcome: Optional[OutcomeRecord] = None

    screenshots: dict[str, str] = Field(
        default_factory=dict,
        description="The files this step wrote, by what each one shows.",
    )
    perceive_ms: Optional[float] = None
    plan_ms: Optional[float] = None
    act_ms: Optional[float] = None

    calls: List[CallRow] = Field(
        default_factory=list, description="The model calls this step paid for."
    )


def _plain(value: Optional[str]) -> str:
    """What a log with no scrubber does, which is what a test gets."""
    return value or ""


class StepLog:
    """Where the steps of one run are written down, as each one happens.

    One owner for how a step is recorded, so the loop reads as the five
    things it does rather than as the record it keeps. Every method here acts
    on the step that is open, and does nothing before the first one is.

    The scrubber is passed in. Every prose field a planner writes goes
    through it, because all of them are persisted and any of them can quote a
    typed value back.
    """

    def __init__(self, scrub: Callable[[Optional[str]], str] = _plain):
        self.records: List[StepRecord] = []
        self._scrub = scrub

    def __len__(self) -> int:
        return len(self.records)

    def open(self, n: int) -> StepRecord:
        """Starts the record for step `n`, counted from one."""
        step_record = StepRecord(n=n)
        self.records.append(step_record)
        return step_record

    @property
    def current(self) -> Optional[StepRecord]:
        return self.records[-1] if self.records else None

    def files(self, screenshot_files: dict) -> None:
        """Notes the files a step wrote, skipping the ones it did not write."""
        step = self.current
        if step is not None:
            step.screenshots.update(
                {name: value for name, value in screenshot_files.items() if value}
            )

    def observed(self, phash: Optional[str], took_ms: float) -> None:
        step = self.current
        if step is not None:
            step.phash = phash
            step.perceive_ms = took_ms

    def planned(self, response, took_ms: float) -> None:
        """Keeps all five of the planner's prose fields, not the reasoning alone.

        The screen name is prose too. A model names a screen from what is on
        it. So a screen that greets an account by name, or shows a value in
        its title, puts that value in the name it is given.
        """
        step = self.current
        if step is None:
            return
        step.plan_ms = took_ms
        screen_name = self._scrub(response.current_screen)
        step.screen_name = screen_name
        step.decision = DecisionRecord(
            current_screen=screen_name,
            evaluation_previous_goal=self._scrub(response.evaluation_previous_goal),
            observation=self._scrub(response.observation),
            reasoning=self._scrub(response.reasoning),
            goal_satisfied=response.goal_satisfied,
            goal_evidence=self._scrub(response.goal_evidence),
        )

    def record_grounding(self, mode: str, decision, requested) -> None:
        """What the planner asked for, and whether the screen could place it."""
        step = self.current
        if step is not None:
            step.grounding = GroundingRecord(
                mode=mode,
                resolved=decision.ok,
                mark_id=requested.model_dump().get("mark"),
                refusal=None if decision.ok else self._scrub(decision.observation),
                unmatched=decision.unmatched,
            )

    def record_action(
        self,
        action: PlannedAction,
        description: str,
        result,
        target: Optional[str],
    ) -> None:
        """What the step did, and what the device said about it.

        The target label is what the element said, which is app content. A
        button labelled with the value it will submit carries that value.
        """
        step = self.current
        if step is None:
            return
        step.action = ActionRecord.of(
            action,
            self._scrub(description),
            self._scrub(target) or None,
            self._scrub(action.reason) or None,
        )
        step.outcome = OutcomeRecord(
            success=result.success,
            observation=self._scrub(result.observation),
            error=self._scrub(result.error) or None,
        )

    def record_blocked_action(
        self,
        action: PlannedAction,
        description: str,
        observation: Optional[str],
        target: Optional[str] = None,
    ) -> None:
        """An action that never reached the device still happened to the step.

        The refusal is the error as well as the observation. A reader
        filtering on `error` would otherwise see a successful-looking step
        that did nothing.
        """
        step = self.current
        if step is None:
            return
        step.action = ActionRecord.of(
            action,
            self._scrub(description),
            self._scrub(target) or None,
            self._scrub(action.reason) or None,
        )
        step.outcome = OutcomeRecord(
            success=False,
            observation=self._scrub(observation),
            error=self._scrub(observation) or None,
        )

    def record_action_duration(self, duration_ms: float) -> None:
        step = self.current
        if step is not None:
            step.act_ms = duration_ms
