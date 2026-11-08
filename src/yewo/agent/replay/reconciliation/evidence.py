"""Build deterministic reconciliation evidence from canonical run records."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from yewo import observations as observation_records
from yewo.agent.actuation import ActionKind, ActionOutcomeDisposition
from yewo.agent.runtime.records import AgentStep, AgentTask
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.agent.runtime.observation_settlement import latest_settlement_candidate
from yewo.agent.replay.reconciliation.records import (
    ControlInference,
    ControlInputTraits,
    ControlInferenceInput,
    ControlValueType,
    EvidenceControl,
    EvidenceDocument,
    EvidenceResult,
    EvidenceRun,
    EvidenceState,
    EvidenceStep,
    EvidenceStepKind,
    EvidenceTask,
    TimelineEntry,
)
from yewo.agent.replay.reconciliation.diagnostics import build_evidence_diagnostics
from yewo.agent.replay.visual_matching import (
    build_observation_visual_signature,
    build_target_visual_descriptor,
)
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.grounding import GroundedPoint, ScreenAnalysisGroundingResolution
from yewo.observations import ScreenComparisonPolicy, ScreenMatcher


_VALUE_TYPE_BY_SEMANTIC_ROLE = {
    ScreenSemanticRole.CHECKBOX: ControlValueType.BOOLEAN,
    ScreenSemanticRole.RADIO_BUTTON: ControlValueType.BOOLEAN,
    ScreenSemanticRole.SWITCH: ControlValueType.BOOLEAN,
    ScreenSemanticRole.DROPDOWN: ControlValueType.ENUM,
    ScreenSemanticRole.OPTION: ControlValueType.ENUM,
    ScreenSemanticRole.SLIDER: ControlValueType.RANGE,
    ScreenSemanticRole.DATE_PICKER: ControlValueType.DATE,
    ScreenSemanticRole.TIME_PICKER: ControlValueType.TIME,
}


class EvidenceConstructionError(ValueError):
    """Canonical records cannot form one coherent evidence document."""


def build_evidence_document(
    *,
    run_id: RecordIdentifier,
    objective_id: RecordIdentifier,
    task: AgentTask,
    steps: Sequence[AgentStep],
    observations: Sequence[observation_records.Observation],
    screen_analyses: Sequence[CurrentScreenAnalysis] = (),
    comparison: ScreenComparisonPolicy | None = None,
    screenshot_loader: Callable[[observation_records.ScreenshotReference], bytes] | None = None,
    identifier_factory: Callable[[], str] = new_record_identifier,
) -> EvidenceDocument:
    """Build one complete, immutable view without dropping source steps."""
    if not observations:
        raise EvidenceConstructionError("Evidence construction requires an observation.")
    policy = comparison or ScreenComparisonPolicy()
    if len({step.step_id for step in steps}) != len(steps):
        raise EvidenceConstructionError("Source step identifiers must be unique.")
    observation_by_id = _index_observations(observations)
    analysis_by_observation = {item.observation_id: item for item in screen_analyses}
    analysis_by_id = {item.analysis_id: item for item in screen_analyses}
    application = observations[0].application
    if any(observation.application != application for observation in observations):
        raise EvidenceConstructionError("All observations must use one application build.")
    device = observations[0].device
    if any(observation.device != device for observation in observations):
        raise EvidenceConstructionError("All observations must use one device identity.")

    states, state_by_observation = _group_states(
        observations,
        policy,
        screenshot_loader,
    )
    evidence_steps = tuple(
        _build_step(
            index,
            step,
            observation_by_id,
            analysis_by_observation,
            analysis_by_id,
            state_by_observation,
            screenshot_loader,
        )
        for index, step in enumerate(steps, start=1)
    )
    run = EvidenceRun(
        run_id=run_id,
        application=application,
        device=device,
        step_count=len(evidence_steps),
        observation_count=len(observations),
        launch_state_id=state_by_observation[observations[0].observation_id],
    )
    evidence_task = EvidenceTask(
        objective_id=objective_id,
        objective=task.objective,
        open_work=task.open_work,
        completion_conditions=task.completion_conditions,
    )
    return EvidenceDocument(
        evidence_id=identifier_factory(),
        run=run,
        task=evidence_task,
        state_grouping_maximum_phash_distance=policy.maximum_replay_phash_distance,
        states=states,
        steps=evidence_steps,
        diagnostics=build_evidence_diagnostics(evidence_steps),
        timeline=tuple(
            TimelineEntry(
                step=step.step,
                source_state_id=step.source_state_id,
                destination_state_id=step.destination_state_id,
            )
            for step in evidence_steps
        ),
    )


def _index_observations(
    observations: Sequence[observation_records.Observation],
) -> dict[str, observation_records.Observation]:
    indexed: dict[str, observation_records.Observation] = {}
    for observation in observations:
        if observation.observation_id in indexed:
            raise EvidenceConstructionError("Observation identifiers must be unique.")
        indexed[observation.observation_id] = observation
    return indexed


def _group_states(
    observations: Sequence[observation_records.Observation],
    policy: ScreenComparisonPolicy,
    screenshot_loader: Callable[[observation_records.ScreenshotReference], bytes] | None,
) -> tuple[dict[str, EvidenceState], dict[str, str]]:
    grouped: list[list[observation_records.Observation]] = []
    for observation in observations:
        matching = next(
            (members for members in grouped if _replay_equivalent(observation, members[0], policy)),
            None,
        )
        if matching is None:
            grouped.append([observation])
        else:
            matching.append(observation)

    states: dict[str, EvidenceState] = {}
    state_by_observation: dict[str, str] = {}
    for ordinal, members in enumerate(grouped, start=1):
        state_id = f"state_{ordinal:03d}"
        representative = members[0]
        visual_signature = build_observation_visual_signature(members, screenshot_loader)
        states[state_id] = EvidenceState(
            state_id=state_id,
            representative_observation_id=representative.observation_id,
            representative_screenshot=representative.redacted_screenshot,
            observation_ids=tuple(item.observation_id for item in members),
            presentations=tuple(item.presentation_key for item in members),
            visual_signature=visual_signature,
        )
        for item in members:
            state_by_observation[item.observation_id] = state_id
    return states, state_by_observation


def _replay_equivalent(
    left: observation_records.Observation,
    right: observation_records.Observation,
    policy: ScreenComparisonPolicy,
) -> bool:
    return ScreenMatcher(policy).replay_equivalent(
        left.presentation_key,
        right.presentation_key,
    )


def _build_step(
    selector: int,
    step: AgentStep,
    observations: dict[str, observation_records.Observation],
    analyses_by_observation: dict[str, CurrentScreenAnalysis],
    analyses_by_id: dict[str, CurrentScreenAnalysis],
    state_by_observation: dict[str, str],
    screenshot_loader: Callable[[observation_records.ScreenshotReference], bytes] | None,
) -> EvidenceStep:
    source = observations.get(step.source_observation_id)
    if source is None:
        raise EvidenceConstructionError("A step source observation is missing.")
    recovered_destination_id = _recovered_settlement_observation_id(
        step,
        source,
        observations,
    )
    destination_id = (
        step.settled_observation_id or step.recovery_observation_id or recovered_destination_id
    )
    if destination_id is not None and destination_id not in observations:
        raise EvidenceConstructionError("A step destination observation is missing.")
    kind = _step_kind(step, source)
    target_visual_descriptor = None
    target_result = step.target_grounding.result if step.target_grounding is not None else None
    if (
        screenshot_loader is not None
        and target_result is not None
        and isinstance(target_result.location, GroundedPoint)
    ):
        screenshot = source.redacted_screenshot
        try:
            png = screenshot_loader(screenshot)
        except observation_records.ObservationCaptureError:
            png = None
        if png is not None:
            target_visual_descriptor = build_target_visual_descriptor(
                png,
                target_result.location.pixel,
                screenshot.width_px,
                screenshot.height_px,
            )
    return EvidenceStep(
        step=selector,
        source_step_id=step.step_id,
        source_observation_id=step.source_observation_id,
        destination_observation_id=destination_id,
        source_state_id=state_by_observation[step.source_observation_id],
        destination_state_id=(
            state_by_observation[destination_id] if destination_id is not None else None
        ),
        kind=kind,
        proposal=step.proposal,
        control=_infer_control(
            step,
            _analysis_for_step(step, analyses_by_observation, analyses_by_id),
        ),
        result=_step_result(step, recovered_destination_id is not None),
        outcome_disposition=(step.outcome.disposition if step.outcome is not None else None),
        refusal_code=(step.refusal.code if step.refusal is not None else None),
        signals=(step.outcome.signals if step.outcome is not None else frozenset()),
        text_delivery=(step.outcome.text_delivery if step.outcome is not None else None),
        authorization_side_effect=(
            step.authorization.side_effect if step.authorization is not None else None
        ),
        target_grounding=step.target_grounding,
        target_visual_descriptor=target_visual_descriptor,
        destination_grounding=step.destination_grounding,
    )


def _analysis_for_step(
    step: AgentStep,
    analyses_by_observation: dict[str, CurrentScreenAnalysis],
    analyses_by_id: dict[str, CurrentScreenAnalysis],
) -> CurrentScreenAnalysis | None:
    grounding = step.target_grounding
    if grounding is not None and isinstance(
        grounding.result.resolution,
        ScreenAnalysisGroundingResolution,
    ):
        return analyses_by_id.get(grounding.result.resolution.analysis_id)
    return analyses_by_observation.get(step.source_observation_id)


def _recovered_settlement_observation_id(
    step: AgentStep,
    source: observation_records.Observation,
    observations: dict[str, observation_records.Observation],
) -> str | None:
    if step.outcome is None or step.outcome.disposition is not ActionOutcomeDisposition.UNAVAILABLE:
        return None
    captured_labels = [
        observations[observation_id]
        for observation_id in step.settlement_observation_ids
        if observation_id in observations
    ]
    for index in range(1, len(captured_labels) + 1):
        candidate = latest_settlement_candidate(
            source,
            captured_labels[:index],
            final_attempt=index == len(captured_labels),
        )
        if candidate is not None:
            return candidate.observation_id
    return None


def _step_kind(step: AgentStep, source: observation_records.Observation) -> EvidenceStepKind:
    action = step.proposal.action
    foreign_context = (
        source.foreground_application_id is not None
        and source.foreground_application_id != source.application.application_id
    )
    if foreign_context or action in {ActionKind.WAIT, ActionKind.SET_DEVICE_CAPABILITY}:
        return EvidenceStepKind.ENVIRONMENT
    return EvidenceStepKind.PATH


def _step_result(step: AgentStep, recovered_destination: bool = False) -> EvidenceResult:
    if step.application_failure is not None:
        return EvidenceResult.CRASHED
    if step.refusal is not None:
        return EvidenceResult.REFUSED
    if step.terminal_reason is not None:
        return EvidenceResult.TERMINAL
    if step.outcome is None:
        return EvidenceResult.UNKNOWN
    if recovered_destination:
        return EvidenceResult.RECOVERED
    return {
        ActionOutcomeDisposition.OBSERVED: EvidenceResult.OBSERVED,
        ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE: EvidenceResult.NO_EFFECT,
        ActionOutcomeDisposition.AMBIGUOUS: EvidenceResult.AMBIGUOUS,
        ActionOutcomeDisposition.UNAVAILABLE: EvidenceResult.NOT_CAPTURED,
    }[step.outcome.disposition]


def _infer_control(
    step: AgentStep,
    analysis: CurrentScreenAnalysis | None,
) -> EvidenceControl:
    analyzed_element = _analyzed_target(step, analysis)
    kind = (
        analyzed_element.kind
        if analyzed_element is not None
        else _control_kind(step.proposal.action)
    )
    label = analyzed_element.name if analyzed_element is not None else None
    value_type = _value_type(kind.semantic)
    identity = _control_identity(kind.semantic, label)
    input_sources: list[ControlInferenceInput] = ["recorded_action"]
    if analyzed_element is not None:
        input_sources.append("current_screen_analysis")
    if step.proposal.text is not None:
        input_sources.append("recorded_input_contract")
    return EvidenceControl(
        kind=kind,
        kind_source="screen_analysis" if analyzed_element is not None else "recorded_action",
        workflow_role=(analyzed_element.workflow_role if analyzed_element is not None else None),
        label=label,
        value_type=value_type,
        identity_key=identity,
        input_traits=ControlInputTraits.from_delivery(
            step.outcome.text_delivery if step.outcome else None
        ),
        inference=ControlInference(
            method=(
                "screen-analysis-element-v1"
                if analyzed_element is not None
                else "recorded-action-kind-v1"
            ),
            input_sources=tuple(input_sources),
            certainty=(
                "inferred"
                if analyzed_element is not None
                or step.proposal.action in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}
                else "unknown"
            ),
        ),
    )


def _analyzed_target(
    step: AgentStep,
    analysis: CurrentScreenAnalysis | None,
) -> ScreenElement | None:
    element_ref = step.proposal.target_element_ref
    if analysis is None or element_ref is None:
        return None
    return next((item for item in analysis.elements if item.element_ref == element_ref), None)


def _control_kind(action: ActionKind) -> ScreenElementKind:
    if action in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}:
        return ScreenElementKind(
            visual=ScreenVisualKind.INPUT_SURFACE,
            semantic=ScreenSemanticRole.TEXT_FIELD,
        )
    return ScreenElementKind(
        visual=ScreenVisualKind.UNKNOWN,
        semantic=ScreenSemanticRole.UNKNOWN,
    )


def _value_type(semantic_role: ScreenSemanticRole) -> ControlValueType:
    inferred_type = _VALUE_TYPE_BY_SEMANTIC_ROLE.get(semantic_role)
    if inferred_type is not None:
        return inferred_type
    if semantic_role is not ScreenSemanticRole.TEXT_FIELD:
        return ControlValueType.NONE
    return ControlValueType.TEXT


def _control_identity(
    semantic_role: ScreenSemanticRole,
    label: str | None,
) -> str | None:
    if label is None:
        return None
    normalized_label = " ".join(label.casefold().split())
    return f"{semantic_role.value}:{normalized_label}" if normalized_label else None
