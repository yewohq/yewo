"""Small deterministic builders for actuation tests."""

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionHandling,
    ActionKind,
    ActionPolicyLimits,
    ActionProposal,
    LiteralText,
    SideEffectClass,
)
from yewo.grounding import (
    GroundedPoint,
    GroundedRegion,
    GroundingDisposition,
    GroundingEvidence,
    GroundingOutputShape,
    GroundingResult,
    GroundingTargetOperation,
    NormalizedPoint,
    NormalizedRegion,
    PixelPoint,
)
from yewo.models import ModelCallDisposition, ModelCallUsage
from yewo.observations import PixelRegion

from grounding.support import build_call_artifact, build_grounding_request


EXECUTABLE_ACTIONS = frozenset(
    action
    for action, definition in ACTION_DEFINITIONS.items()
    if definition.handling is ActionHandling.EXECUTE
)


def limits(**updates: object) -> ActionPolicyLimits:
    values: dict[str, object] = {
        "allowed_actions": EXECUTABLE_ACTIONS,
        "supported_actions": EXECUTABLE_ACTIONS,
        "allowed_side_effects": frozenset(SideEffectClass),
        "maximum_actions": 20,
    }
    values.update(updates)
    return ActionPolicyLimits.model_validate(values)


def budget(**updates: object) -> ActionBudget:
    values: dict[str, object] = {"dispatched_actions": 0}
    values.update(updates)
    return ActionBudget.model_validate(values)


def proposal(action: ActionKind = ActionKind.TAP, **updates: object) -> ActionProposal:
    values: dict[str, object] = {
        "proposal_id": "proposal-1",
        "source_observation_id": "obs_1",
        "action": action,
    }
    defaults: dict[ActionKind, dict[str, object]] = {
        ActionKind.TAP: {
            "target_expression": "the add button",
            "expected_outcome": "The editor opens.",
        },
        ActionKind.FOCUS: {
            "target_expression": "the name field",
            "expected_outcome": "The name field receives focus.",
        },
        ActionKind.TYPE: {
            "text": LiteralText(value="hello"),
            "expected_outcome": "The name field contains the text.",
        },
        ActionKind.CLEAR: {
            "expected_outcome": "The name field is empty.",
        },
        ActionKind.DOUBLE_TAP: {
            "target_expression": "the image",
            "expected_outcome": "The image enlarges.",
        },
        ActionKind.LONG_PRESS: {
            "target_expression": "the item",
            "duration_seconds": 1,
            "expected_outcome": "The item menu opens.",
        },
        ActionKind.DRAG: {
            "target_expression": "the first item",
            "destination_expression": "the second position",
            "duration_seconds": 1,
            "expected_outcome": "The item moves.",
        },
        ActionKind.SCROLL: {
            "direction": "down",
            "distance": "short",
            "expected_outcome": "Lower content becomes visible.",
        },
        ActionKind.BACK: {"expected_outcome": "The previous screen appears."},
        ActionKind.DISMISS_KEYBOARD: {"expected_outcome": "The keyboard closes."},
        ActionKind.WAIT: {
            "duration_seconds": 0.5,
            "expected_outcome": "Transient rendering finishes.",
        },
        ActionKind.SET_DEVICE_CAPABILITY: {"reason": "The capability suite owns this request."},
        ActionKind.DONE: {"reason": "Task is complete."},
        ActionKind.CANNOT_PROGRESS: {"reason": "No permitted action can progress."},
    }
    values.update(defaults[action])
    if action not in {
        ActionKind.SET_DEVICE_CAPABILITY,
        ActionKind.DONE,
        ActionKind.CANNOT_PROGRESS,
    }:
        values["reason"] = "This is the next useful action supported by current evidence."
    values.update(updates)
    return ActionProposal.model_validate(values)


def point_grounding(
    *,
    grounding_id: str = "grounding-1",
    screenshot_sha256: str = "a" * 64,
    expression: str = "the add button",
    operation: GroundingTargetOperation = GroundingTargetOperation.TAP,
    x: int = 100,
    y: int = 200,
    disposition: GroundingDisposition = GroundingDisposition.ACCEPTED,
    transport_disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
    reason: str = "No target found.",
) -> GroundingEvidence:
    request = build_grounding_request(
        grounding_id=grounding_id,
        expression=expression,
        operation=operation,
        output_shape=GroundingOutputShape.POINT,
    )
    if screenshot_sha256 != request.screenshot.sha256:
        request = request.model_copy(
            update={
                "screenshot": request.screenshot.model_copy(update={"sha256": screenshot_sha256})
            }
        )
    location = (
        GroundedPoint(
            normalized=NormalizedPoint(x=x / 199, y=y / 399),
            pixel=PixelPoint(x=x, y=y),
        )
        if disposition is GroundingDisposition.ACCEPTED
        else None
    )
    return GroundingEvidence(
        request=request,
        result=_grounding(
            grounding_id,
            screenshot_sha256,
            GroundingOutputShape.POINT,
            disposition,
            location,
            transport_disposition,
            reason,
        ),
    )


def region_grounding(
    *,
    grounding_id: str = "grounding-1",
    expression: str = "the results list",
) -> GroundingEvidence:
    request = build_grounding_request(
        grounding_id=grounding_id,
        expression=expression,
        operation=GroundingTargetOperation.SCROLL_REGION,
        output_shape=GroundingOutputShape.REGION,
    )
    return GroundingEvidence(
        request=request,
        result=_grounding(
            grounding_id,
            request.screenshot.sha256,
            GroundingOutputShape.REGION,
            GroundingDisposition.ACCEPTED,
            GroundedRegion(
                normalized=NormalizedRegion(left=0.1, top=0.2, right=0.9, bottom=0.8),
                pixel=PixelRegion(left=20, top=80, right=180, bottom=320),
            ),
        ),
    )


def _grounding(
    grounding_id: str,
    screenshot_sha256: str,
    output_shape: GroundingOutputShape,
    disposition: GroundingDisposition,
    location: GroundedPoint | GroundedRegion | None,
    transport_disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
    reason: str = "No target found.",
) -> GroundingResult:
    return GroundingResult(
        grounding_id=grounding_id,
        profile_id="grounder",
        model_profile_id="grounder-model",
        source_screenshot_sha256=screenshot_sha256,
        output_shape=output_shape,
        disposition=disposition,
        location=location,
        reason=None if disposition is GroundingDisposition.ACCEPTED else reason,
        requested_provider="offline",
        requested_model="test-grounder",
        transport_disposition=transport_disposition,
        raw_response_artifact=build_call_artifact(),
        usage=ModelCallUsage(),
        model_request_count=1,
    )
