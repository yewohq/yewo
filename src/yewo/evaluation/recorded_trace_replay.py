"""Replay reviewed run fragments through deterministic Yewo components."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field, model_validator

from yewo.agent.actuation.actions import Coordinates, PlannerTapAction
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory.history import ActionEntry, ActionResult, AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.perception.keyboard import observe
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.run.recovery import (
    RecoveryCause,
    RecoveryTransaction,
)
from yewo.agent.run.instruction_controller import (
    InstructionCompletionReceipt,
    InstructionController,
)
from yewo.testing.evidence import AssertionScope, ExpectedOutcome
from yewo.testing.judge import rules
from yewo.testing.obligations import (
    EvidenceOwner,
    ObligationActivation,
    ObligationContract,
    ObligationController,
    ObligationEvent,
    ObligationKind,
    ObligationReadiness,
    ObligationSettlement,
    aggregate_readiness,
    select_outcomes,
)

DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "eval"
    / "recorded_traces"
    / "trainer_app_1645_buggy.json"
)


class SourceArtifact(BaseModel):
    """Immutable provenance for evidence copied out of an ignored run."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_path(self) -> "SourceArtifact":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("A source artifact path must be repository-relative.")
        return self


class RunBaseline(BaseModel):
    """Reviewed cost and outcome measurements from one recorded scenario."""

    run_id: str
    scenario_status: str
    steps_executed: int = Field(ge=0)
    scenario_duration_seconds: float = Field(ge=0)
    failure_reason: str
    issues_found: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    model_latency_ms: float = Field(ge=0)
    time_to_first_finding_ms: float | None = Field(default=None, ge=0)


class LoopActionEvent(BaseModel):
    event: Literal["action"]
    screen_hash: str
    x: int
    y: int
    target: str
    screen_size: tuple[int, int] = (1080, 2400)


class LoopDeliveryEvent(BaseModel):
    event: Literal["delivery"]
    before_hash: str
    after_hash: str
    x: int
    y: int
    target: str
    failed: bool = False
    screen_size: tuple[int, int] = (1080, 2400)


class LoopScreenEvent(BaseModel):
    event: Literal["screen"]
    screen_hash: str


class LoopStaleEvent(BaseModel):
    event: Literal["stale"]
    screen_hash: str
    x: int
    y: int
    target: str
    screen_size: tuple[int, int] = (1080, 2400)


LoopEvent = Annotated[
    LoopActionEvent | LoopDeliveryEvent | LoopScreenEvent | LoopStaleEvent,
    Field(discriminator="event"),
]


class LoopReading(BaseModel):
    action_blocked: bool
    terminated: bool
    last_message: str
    termination_reason: str


class LoopCase(BaseModel):
    kind: Literal["loop"]
    id: str
    why: str
    source_run_id: str
    events: list[LoopEvent]
    baseline: LoopReading
    required: LoopReading
    known_gap: bool


class KeyboardReading(BaseModel):
    native_visible: bool | None
    visual_visible: bool
    effective_visible: bool
    state_disagrees: bool


class KeyboardCase(BaseModel):
    kind: Literal["keyboard"]
    id: str
    why: str
    source_run_id: str
    native_visible: bool | None
    visual_keyboard_present: bool
    baseline: KeyboardReading
    required: KeyboardReading
    known_gap: bool


class InstructionReading(BaseModel):
    active_instruction_index: int | None
    active_instruction: str | None
    recovery_status: str
    recovery_active: bool


class InstructionCase(BaseModel):
    kind: Literal["instruction"]
    id: str
    why: str
    source_run_id: str
    instructions: list[str]
    initial_instruction_index: int
    model_instruction_index: int
    model_claimed_goal_satisfied: bool
    baseline: InstructionReading
    required: InstructionReading
    known_gap: bool


class HistoryReading(BaseModel):
    self_assessment_in_prompt: bool
    disputed_phrase_in_prompt: bool


class HistoryCase(BaseModel):
    kind: Literal["history"]
    id: str
    why: str
    source_run_id: str
    action_description: str
    device_observation: str
    model_evaluation: str
    disputed_phrase: str
    baseline: HistoryReading
    required: HistoryReading
    known_gap: bool


class InputReading(BaseModel):
    value_verified: bool
    context_retained: bool
    input_complete: bool


class InputCase(BaseModel):
    kind: Literal["input"]
    id: str
    why: str
    source_run_id: str
    dispatch_success: bool
    value_verified: bool
    before_context: str
    after_context: str
    baseline: InputReading
    required: InputReading
    known_gap: bool


class AuditorReading(BaseModel):
    scenario_rule_sent: bool


class AuditorCase(BaseModel):
    kind: Literal["auditor"]
    id: str
    why: str
    source_run_id: str
    rule_statement: str
    rule_scope: AssertionScope
    evidence_phase: Literal["intermediate", "terminal"]
    baseline: AuditorReading
    required: AuditorReading
    known_gap: bool


class TerminalReading(BaseModel):
    last_milestone: str
    terminal_reason: str


class TerminalCase(BaseModel):
    kind: Literal["terminal"]
    id: str
    why: str
    source_run_id: str
    milestones: list[str]
    terminal_reason: str
    baseline: TerminalReading
    required: TerminalReading
    known_gap: bool


TraceCase = Annotated[
    LoopCase
    | KeyboardCase
    | InstructionCase
    | HistoryCase
    | InputCase
    | AuditorCase
    | TerminalCase,
    Field(discriminator="kind"),
]


class TraceCorpus(BaseModel):
    """Versioned reviewed traces and the baseline they currently reproduce."""

    schema_version: Literal[1]
    app_id: str
    reviewed_at: str
    sources: list[SourceArtifact]
    run_baselines: list[RunBaseline]
    cases: list[TraceCase]

    @model_validator(mode="after")
    def _validate_cases(self) -> "TraceCorpus":
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Recorded trace case identifiers must be unique.")
        source_runs = {baseline.run_id for baseline in self.run_baselines}
        for case in self.cases:
            if case.source_run_id not in source_runs:
                raise ValueError(f"Trace {case.id} names a run without a baseline.")
            differs = case.baseline.model_dump() != case.required.model_dump()
            if differs != case.known_gap:
                raise ValueError(f"Trace {case.id} has an inconsistent known-gap flag.")
        return self


class ReplayResult(BaseModel):
    case_id: str
    matches_baseline: bool
    known_gap: bool
    observed: dict[str, object]


def load_corpus(path: Path = DEFAULT_CORPUS) -> TraceCorpus:
    return TraceCorpus.model_validate_json(path.read_bytes())


def _tap(x: int, y: int, target: str) -> PlannerTapAction:
    action = TapCoordinateAction(
        coordinates=Coordinates(x=x, y=y), target=target, reason="replay trace"
    )
    return cast(PlannerTapAction, action)


def _replay_loop(case: LoopCase) -> LoopReading:
    detector = RollingLoopDetector()
    last_message = ""
    for event in case.events:
        if isinstance(event, LoopActionEvent):
            message = detector.record_action(
                _tap(event.x, event.y, event.target),
                event.screen_hash,
                event.screen_size,
            )
        elif isinstance(event, LoopDeliveryEvent):
            message = detector.record_delivery_outcome(
                _tap(event.x, event.y, event.target),
                event.before_hash,
                event.after_hash,
                event.screen_size,
                failed=event.failed,
            )
        elif isinstance(event, LoopScreenEvent):
            message = detector.record_screen(event.screen_hash)
        else:
            message = detector.record_stale(
                _tap(event.x, event.y, event.target),
                event.screen_hash,
                event.screen_size,
            )
        if message:
            last_message = message
    return LoopReading(
        action_blocked=detector.action_blocked,
        terminated=detector.terminated(),
        last_message=last_message,
        termination_reason=detector.termination_reason or "",
    )


def _replay_keyboard(case: KeyboardCase) -> KeyboardReading:
    elements: list[GUIElement] = []
    if case.visual_keyboard_present:
        elements.append(
            GUIElement(
                element="recorded-keyboard-region",
                bounding_box=BoundingBox(x_min=0, y_min=1200, x_max=1080, y_max=2400),
                confidence=1.0,
                type="region",
                semantic_roles=frozenset({"keyboard_region"}),
            )
        )
    reading = observe(elements, native_visible=case.native_visible)
    return KeyboardReading(
        native_visible=reading.native_visible,
        visual_visible=reading.visual_visible,
        effective_visible=reading.effective_visible,
        state_disagrees=reading.state_disagrees,
    )


def _replay_instruction(case: InstructionCase) -> InstructionReading:
    recovery = RecoveryTransaction.open(
        failed_step=7,
        opened_step=8,
        original_intent=case.instructions[case.initial_instruction_index - 1],
        failure_reason="The recorded route returned to an earlier state.",
        cause=RecoveryCause.SEMANTIC_OBSTRUCTION,
    )
    controller = InstructionController.from_goal(case.instructions)
    for completed_index in range(1, case.initial_instruction_index):
        instruction = controller.active_instruction
        if instruction is None:
            raise ValueError("The recorded instruction prefix is inconsistent.")
        controller.accept_completion(
            InstructionCompletionReceipt(
                instruction_id=instruction.instruction_id,
                observation_id=f"recorded-observation-{completed_index}",
                evidence_owner="recorded-trace-review",
            )
        )
    controller.record_planner_claim(
        case.model_instruction_index,
        case.model_claimed_goal_satisfied,
    )
    active = controller.active_instruction
    return InstructionReading(
        active_instruction_index=controller.active_index,
        active_instruction=active.intent if active is not None else None,
        recovery_status=recovery.status.value,
        recovery_active=True,
    )


def _replay_history(case: HistoryCase) -> HistoryReading:
    history = AgentHistory()
    history.append(
        ActionEntry(
            step_num=1,
            action_desc=case.action_description,
            result=ActionResult(success=True, observation=case.device_observation),
        )
    )
    prompt_history = history.format_for_prompt()
    return HistoryReading(
        self_assessment_in_prompt="Self-assessment:" in prompt_history,
        disputed_phrase_in_prompt=case.disputed_phrase in prompt_history,
    )


def _replay_input(case: InputCase) -> InputReading:
    context_retained = case.before_context == case.after_context
    return InputReading(
        value_verified=case.value_verified,
        context_retained=context_retained,
        input_complete=(
            case.dispatch_success and case.value_verified and context_retained
        ),
    )


def _replay_auditor(case: AuditorCase) -> AuditorReading:
    outcome = ExpectedOutcome(statement=case.rule_statement, scope=case.rule_scope)
    controller = _auditor_controller()
    indexes = controller.route(_auditor_event(case))
    routed = select_outcomes([outcome], indexes)
    listed = rules.every_rule(rules.on_screen_only(routed), [])
    return AuditorReading(scenario_rule_sent=case.rule_statement in listed)


def _auditor_controller() -> ObligationController:
    return ObligationController(
        [
            ObligationContract(
                obligation_id="recorded-outcome",
                outcome_index=0,
                kind=ObligationKind.SCREEN_ASSERTION,
                evidence_owner=EvidenceOwner.SCREEN_VERIFIER,
                activation=ObligationActivation.AT_RUN_END,
                settlement=ObligationSettlement.EXISTS,
                required=True,
            )
        ]
    )


def _auditor_event(case: AuditorCase) -> ObligationEvent:
    return (
        ObligationEvent.RUN_END
        if case.evidence_phase == "terminal"
        else ObligationEvent.OBSERVATION
    )


def corpus_obligation_readiness(corpus: TraceCorpus) -> ObligationReadiness:
    """Measure strict-routing readiness across reviewed semantic trace cases."""
    measurements = []
    for case in corpus.cases:
        if not isinstance(case, AuditorCase):
            continue
        controller = _auditor_controller()
        controller.route(_auditor_event(case))
        measurements.append(controller.readiness())
    return aggregate_readiness(measurements)


def _replay_terminal(case: TerminalCase) -> TerminalReading:
    return TerminalReading(
        last_milestone=case.milestones[-1], terminal_reason=case.terminal_reason
    )


def replay_case(case: TraceCase) -> ReplayResult:
    reading: (
        LoopReading
        | KeyboardReading
        | InstructionReading
        | HistoryReading
        | InputReading
        | AuditorReading
        | TerminalReading
    )
    if isinstance(case, LoopCase):
        reading = _replay_loop(case)
    elif isinstance(case, KeyboardCase):
        reading = _replay_keyboard(case)
    elif isinstance(case, InstructionCase):
        reading = _replay_instruction(case)
    elif isinstance(case, HistoryCase):
        reading = _replay_history(case)
    elif isinstance(case, InputCase):
        reading = _replay_input(case)
    elif isinstance(case, AuditorCase):
        reading = _replay_auditor(case)
    else:
        reading = _replay_terminal(case)
    observed = reading.model_dump()
    return ReplayResult(
        case_id=case.id,
        matches_baseline=observed == case.baseline.model_dump(),
        known_gap=case.known_gap,
        observed=observed,
    )


def replay_corpus(corpus: TraceCorpus) -> list[ReplayResult]:
    return [replay_case(case) for case in corpus.cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    arguments = parser.parse_args()
    results = replay_corpus(load_corpus(arguments.corpus))
    for result in results:
        label = "known gap" if result.known_gap else "contract"
        state = "matched" if result.matches_baseline else "DRIFTED"
        print(f"{result.case_id}: {state} ({label})")
    return 0 if all(result.matches_baseline for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
