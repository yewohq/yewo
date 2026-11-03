"""Build deterministic reconciliation evidence from canonical run records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re

from yewo import observations as observation_records
from yewo.agent.actuation import ActionKind, ActionOutcomeDisposition
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.runtime.records import AgentStep, AgentTask
from yewo.agent.replay.reconciliation.records import (
    ControlInference,
    ControlInputTraits,
    ControlInferenceInput,
    ControlRole,
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
from yewo.grounding import GroundedPoint
from yewo.observations import fingerprint_distance


_QUOTED_TEXT = re.compile(r"['\"]([^'\"]+)['\"]")
_REFERENCE_PREFIX = re.compile(
    r"(?:left of|right of|below|above|under|beside|after|before|showing|"
    r"title|header|containing|of the|in the|on the|screen|dialog|form|list)\s*(?:the\s*)?$",
    re.IGNORECASE,
)
_FIELD_NAME = re.compile(
    r"^(?:focused |empty )?([a-z][\w \[\]-]{1,80}?) "
    r"(?:text |numeric )?(?:input )?field",
    re.IGNORECASE,
)
_COMMIT_WORDS = re.compile(
    r"\b(?:save|ok|submit|confirm|apply|create|add|finish|checkmark)\b",
    re.IGNORECASE,
)
_INTEGER = re.compile(r"^[+-]?\d+$")
_DECIMAL = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)$")


class EvidenceConstructionError(ValueError):
    """Canonical records cannot form one coherent evidence document."""


def build_evidence_document(
    *,
    run_id: RecordIdentifier,
    objective_id: RecordIdentifier,
    task: AgentTask,
    steps: Sequence[AgentStep],
    observations: Sequence[observation_records.Observation],
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
        screenshot = representative.redacted_screenshot
        visual_signature = build_observation_visual_signature(members, screenshot_loader)
        states[state_id] = EvidenceState(
            state_id=state_id,
            representative_observation_id=representative.observation_id,
            observation_ids=tuple(item.observation_id for item in members),
            foreground_application_id=representative.foreground_application_id,
            platform_context=representative.platform_context,
            keyboard_visible=representative.keyboard_visible,
            orientation=representative.orientation,
            width_px=screenshot.width_px,
            height_px=screenshot.height_px,
            fingerprint=representative.screen_fingerprint,
            fingerprints=tuple(item.screen_fingerprint for item in members),
            screenshot_artifact_id=screenshot.artifact_id,
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
    left_shot = left.redacted_screenshot
    right_shot = right.redacted_screenshot
    if (
        left.foreground_application_id != right.foreground_application_id
        or left.platform_context != right.platform_context
        or left.keyboard_visible is not right.keyboard_visible
        or left.orientation is not right.orientation
        or (left_shot.width_px, left_shot.height_px) != (right_shot.width_px, right_shot.height_px)
    ):
        return False
    try:
        return (
            fingerprint_distance(left.screen_fingerprint, right.screen_fingerprint)
            <= policy.maximum_replay_phash_distance
        )
    except ValueError:
        return False


def _build_step(
    selector: int,
    step: AgentStep,
    observations: dict[str, observation_records.Observation],
    state_by_observation: dict[str, str],
    screenshot_loader: Callable[[observation_records.ScreenshotReference], bytes] | None,
) -> EvidenceStep:
    source = observations.get(step.source_observation_id)
    if source is None:
        raise EvidenceConstructionError("A step source observation is missing.")
    destination_id = step.settled_observation_id or step.recovery_observation_id
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
        recorded_step_index=step.step_index,
        source_step_id=step.step_id,
        source_observation_id=step.source_observation_id,
        destination_observation_id=destination_id,
        source_state_id=state_by_observation[step.source_observation_id],
        destination_state_id=(
            state_by_observation[destination_id] if destination_id is not None else None
        ),
        kind=kind,
        action=step.proposal.action,
        proposal=step.proposal,
        control=_infer_control(step),
        result=_step_result(step),
        outcome_disposition=(step.outcome.disposition if step.outcome is not None else None),
        signals=(step.outcome.signals if step.outcome is not None else frozenset()),
        text_delivery=(step.outcome.text_delivery if step.outcome is not None else None),
        authorization_side_effect=(
            step.authorization.side_effect if step.authorization is not None else None
        ),
        target_grounding=step.target_grounding,
        target_visual_descriptor=target_visual_descriptor,
        destination_grounding=step.destination_grounding,
        refused=step.refusal is not None,
        application_failed=step.application_failure is not None,
    )


def _step_kind(step: AgentStep, source: observation_records.Observation) -> EvidenceStepKind:
    action = step.proposal.action
    foreign_context = (
        source.foreground_application_id is not None
        and source.foreground_application_id != source.application.application_id
    )
    if foreign_context or action in {ActionKind.WAIT, ActionKind.SET_DEVICE_CAPABILITY}:
        return EvidenceStepKind.ENVIRONMENT
    return EvidenceStepKind.PATH


def _step_result(step: AgentStep) -> EvidenceResult:
    if step.application_failure is not None:
        return EvidenceResult.CRASHED
    if step.refusal is not None:
        return EvidenceResult.REFUSED
    if step.terminal_reason is not None:
        return EvidenceResult.TERMINAL
    if step.outcome is None:
        return EvidenceResult.UNKNOWN
    return {
        ActionOutcomeDisposition.OBSERVED: EvidenceResult.OBSERVED,
        ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE: EvidenceResult.NO_EFFECT,
        ActionOutcomeDisposition.AMBIGUOUS: EvidenceResult.AMBIGUOUS,
        ActionOutcomeDisposition.UNAVAILABLE: EvidenceResult.NOT_CAPTURED,
    }[step.outcome.disposition]


def _infer_control(step: AgentStep) -> EvidenceControl:
    expression = step.proposal.target_expression
    role = _control_role(step.proposal.action, expression)
    label = _semantic_label(expression)
    value_type = _value_type(step, role, expression)
    identity = _control_identity(role, label, expression)
    input_sources: list[ControlInferenceInput] = ["recorded_action"]
    if expression is not None:
        input_sources.append("model_authored_target_expression")
    if step.proposal.text is not None:
        input_sources.append("recorded_input_contract")
    return EvidenceControl(
        role=role,
        label=label,
        value_type=value_type,
        identity_key=identity,
        input_traits=ControlInputTraits.from_delivery(
            step.outcome.text_delivery if step.outcome else None
        ),
        inference=ControlInference(
            input_sources=tuple(input_sources),
            certainty=(
                "inferred"
                if expression is not None
                or step.proposal.action in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}
                else "unknown"
            ),
        ),
    )


def _control_role(action: ActionKind, expression: str | None) -> ControlRole:
    if action in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}:
        return ControlRole.TEXT_FIELD
    if expression is None:
        return ControlRole.UNKNOWN
    lowered = expression.lower()
    candidates = (
        ("image button", ControlRole.IMAGE_BUTTON),
        ("radio button", ControlRole.RADIO_BUTTON),
        ("date picker", ControlRole.DATE_PICKER),
        ("time picker", ControlRole.TIME_PICKER),
        ("menu item", ControlRole.MENU_ITEM),
        ("list-item", ControlRole.LIST_ITEM),
        ("list item", ControlRole.LIST_ITEM),
        ("static text", ControlRole.STATIC_TEXT),
        ("text field", ControlRole.TEXT_FIELD),
        ("input field", ControlRole.TEXT_FIELD),
        ("checkbox", ControlRole.CHECKBOX),
        ("switch", ControlRole.SWITCH),
        ("slider", ControlRole.SLIDER),
        ("dropdown", ControlRole.DROPDOWN),
        ("option", ControlRole.OPTION),
        ("button", ControlRole.BUTTON),
        ("tab", ControlRole.TAB),
        ("link", ControlRole.LINK),
        ("list", ControlRole.LIST),
        (" row", ControlRole.ROW),
        ("card", ControlRole.CARD),
        ("icon", ControlRole.ICON),
        ("label", ControlRole.STATIC_TEXT),
    )
    matches = ((lowered.find(text), role) for text, role in candidates if text in lowered)
    return min(matches, default=(0, ControlRole.UNKNOWN), key=lambda item: item[0])[1]


def _semantic_label(expression: str | None) -> str | None:
    if expression is None:
        return None
    for match in _QUOTED_TEXT.finditer(expression):
        if not _REFERENCE_PREFIX.search(expression[: match.start()].rstrip()):
            return match.group(1).strip().rstrip(":")
    landmark = re.search(
        r"field\b.*?(?:right of|below|under|next to)\s+(?:the\s+)?"
        r"(?:checked\s+)?['\"]([^'\"]+)['\"]",
        expression,
        re.IGNORECASE,
    )
    if landmark is not None:
        return landmark.group(1).strip().rstrip(":")
    field = _FIELD_NAME.match(expression)
    if field is not None:
        candidate = field.group(1).strip()
        if candidate.lower() not in {"text", "numeric", "input", "empty"}:
            return candidate
    return None


def _value_type(
    step: AgentStep,
    role: ControlRole,
    expression: str | None,
) -> ControlValueType:
    if role in {ControlRole.CHECKBOX, ControlRole.RADIO_BUTTON, ControlRole.SWITCH}:
        return ControlValueType.BOOLEAN
    if role in {ControlRole.DROPDOWN, ControlRole.OPTION}:
        return ControlValueType.ENUM
    if role is ControlRole.SLIDER:
        return ControlValueType.RANGE
    if role is ControlRole.DATE_PICKER:
        return ControlValueType.DATE
    if role is ControlRole.TIME_PICKER:
        return ControlValueType.TIME
    if role is not ControlRole.TEXT_FIELD:
        return ControlValueType.NONE
    text = step.proposal.text
    if text is None or text.source != "literal":
        return ControlValueType.TEXT
    lowered = (expression or "").lower()
    if ("numeric" in lowered or "number" in lowered) and _INTEGER.fullmatch(text.value):
        return ControlValueType.INTEGER
    if ("numeric" in lowered or "number" in lowered) and _DECIMAL.fullmatch(text.value):
        return ControlValueType.DECIMAL
    return ControlValueType.TEXT


def _control_identity(
    role: ControlRole, label: str | None, expression: str | None
) -> str | None:
    source = label or expression
    if source is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    return f"{role.value}:{normalized}" if normalized else None


def is_commit_target(expression: str | None) -> bool:
    """Return whether recorded target wording positively identifies a commit control."""
    return expression is not None and (
        _COMMIT_WORDS.search(expression) is not None
    )
