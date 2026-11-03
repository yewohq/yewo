"""Fail-closed authorization for device action proposals."""

from __future__ import annotations

from collections.abc import Callable

from yewo.artifacts import new_record_identifier
from yewo.devices.application_lifecycle import ApplicationState, trusted_system_surface
from yewo.devices.interaction import (
    MINIMUM_ANDROID_DRAG_SECONDS,
    MINIMUM_IOS_DRAG_SECONDS,
    MINIMUM_LONG_PRESS_SECONDS,
    MINIMUM_WAIT_SECONDS,
)
from yewo.devices.inventory import DevicePlatform
from yewo.grounding import (
    GroundedPoint,
    GroundedRegion,
    GroundingDisposition,
    GroundingEvidence,
    GroundingTargetOperation,
    PixelPoint,
)
from yewo.observations import Observation, PixelRegion, ScreenRegion

from yewo.agent.actuation.records import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionHandling,
    ActionKind,
    ActionPolicyLimits,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    ActionSource,
    AuthorizedAction,
    TapRepetition,
)


class ActionPolicy:
    """Convert current proposal evidence into one authorization."""

    def __init__(
        self,
        limits: ActionPolicyLimits,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._limits = limits
        self._identifier_factory = identifier_factory

    def authorize(
        self,
        proposal: ActionProposal,
        observation: Observation,
        budget: ActionBudget,
        *,
        target_grounding: GroundingEvidence | None = None,
        destination_grounding: GroundingEvidence | None = None,
        source: ActionSource = ActionSource.PLANNER,
        tap_repetition: TapRepetition | None = None,
        expected_back_destination_observation_id: str | None = None,
    ) -> AuthorizedAction | ActionRefusal:
        definition = ACTION_DEFINITIONS[proposal.action]
        refusal = self._check_common(
            proposal,
            observation,
            budget,
            source,
            tap_repetition,
            expected_back_destination_observation_id,
        )
        if refusal is not None:
            return refusal
        if (proposal.target_expression is None) != (target_grounding is None):
            return self._refuse(
                proposal,
                ActionRefusalCode.INVALID_GROUNDING,
                "Target grounding does not match the proposal.",
            )
        if (proposal.destination_expression is None) != (destination_grounding is None):
            return self._refuse(
                proposal,
                ActionRefusalCode.INVALID_GROUNDING,
                "Destination grounding does not match the proposal.",
            )

        target, region, refusal = self._resolve_grounding(
            proposal,
            observation,
            proposal.target_expression,
            definition.target_operation,
            target_grounding,
        )
        if refusal is not None:
            return refusal
        destination, _, refusal = self._resolve_grounding(
            proposal,
            observation,
            proposal.destination_expression,
            definition.destination_operation,
            destination_grounding,
        )
        if refusal is not None:
            return refusal
        if proposal.action is ActionKind.DRAG and target == destination:
            return self._refuse(
                proposal, ActionRefusalCode.INVALID_CONSTRAINT, "Drag points must be distinct."
            )

        text = proposal.text
        sensitivity = text.sensitivity if text is not None else proposal.sensitivity
        retention = text.retention if text is not None else proposal.retention
        if proposal.expected_outcome is None:
            return self._refuse(
                proposal,
                ActionRefusalCode.INVALID_CONSTRAINT,
                "The proposal has no expected outcome.",
            )
        return AuthorizedAction(
            authorization_id=self._identifier_factory(),
            proposal_id=proposal.proposal_id,
            source_observation_id=observation.observation_id,
            source_screenshot_sha256=observation.redacted_screenshot.sha256,
            application_id=observation.application.application_id,
            action=proposal.action,
            source=source,
            side_effect=definition.side_effect,
            expected_outcome=proposal.expected_outcome,
            target=target,
            destination=destination,
            region=region,
            text=text,
            duration_seconds=proposal.duration_seconds,
            direction=proposal.direction,
            distance=proposal.distance,
            tap_count=tap_repetition.count if tap_repetition is not None else 1,
            tap_interval_seconds=(
                tap_repetition.interval_seconds if tap_repetition is not None else 0
            ),
            expected_back_destination_observation_id=(
                expected_back_destination_observation_id
                if proposal.action is ActionKind.BACK
                else None
            ),
            sensitivity=sensitivity,
            retention=retention,
        )

    def refusal_before_grounding(
        self,
        proposal: ActionProposal,
        observation: Observation,
        budget: ActionBudget,
        *,
        source: ActionSource = ActionSource.PLANNER,
        tap_repetition: TapRepetition | None = None,
        expected_back_destination_observation_id: str | None = None,
    ) -> ActionRefusal | None:
        """Reject state-invalid proposals before visual grounding."""
        return self._check_common(
            proposal,
            observation,
            budget,
            source,
            tap_repetition,
            expected_back_destination_observation_id,
        )

    def _check_common(
        self,
        proposal: ActionProposal,
        observation: Observation,
        budget: ActionBudget,
        source: ActionSource,
        tap_repetition: TapRepetition | None,
        expected_back_destination_observation_id: str | None,
    ) -> ActionRefusal | None:
        if proposal.source_observation_id != observation.observation_id:
            return self._refuse(
                proposal, ActionRefusalCode.STALE_OBSERVATION, "Proposal observation is stale."
            )
        definition = ACTION_DEFINITIONS[proposal.action]
        if definition.handling is ActionHandling.TERMINAL:
            return self._refuse(
                proposal,
                ActionRefusalCode.NON_DISPATCH_ACTION,
                "Planner terminal signals are not device actions.",
            )
        if definition.handling is not ActionHandling.EXECUTE:
            return self._refuse(
                proposal,
                ActionRefusalCode.UNSUPPORTED_ACTION,
                "Device capability changes activate with the capability component.",
            )
        if proposal.action not in self._limits.allowed_actions:
            return self._refuse(
                proposal,
                ActionRefusalCode.ACTION_NOT_ALLOWED,
                "Action is not allowed in this mode.",
            )
        side_effect = definition.side_effect
        if side_effect not in self._limits.allowed_side_effects:
            return self._refuse(
                proposal,
                ActionRefusalCode.SIDE_EFFECT_NOT_ALLOWED,
                "Action side effect is not allowed in this mode.",
            )
        if proposal.action not in self._limits.supported_actions:
            return self._refuse(
                proposal,
                ActionRefusalCode.UNSUPPORTED_ACTION,
                "The active device boundary does not support this action.",
            )
        if budget.dispatched_actions >= self._limits.maximum_actions:
            return self._refuse(
                proposal,
                ActionRefusalCode.ACTION_BUDGET_EXHAUSTED,
                "The action budget is exhausted.",
            )
        system_surface = trusted_system_surface(
            observation.device.platform,
            observation.foreground_application_id,
        )
        if (
            system_surface is None
            and observation.application_state is not ApplicationState.FOREGROUND
        ):
            return self._refuse(
                proposal,
                ActionRefusalCode.APPLICATION_SCOPE,
                "The application is not in the foreground.",
            )
        foreground = observation.foreground_application_id
        if (
            system_surface is None
            and foreground is not None
            and foreground != observation.application.application_id
        ):
            return self._refuse(
                proposal,
                ActionRefusalCode.APPLICATION_SCOPE,
                "A different application owns the foreground.",
            )
        refusal = self._check_tap_repetition(proposal, source, tap_repetition)
        if refusal is not None:
            return refusal
        refusal = self._check_keyboard_and_back(
            proposal,
            observation,
            expected_back_destination_observation_id,
        )
        if refusal is not None:
            return refusal
        return self._check_duration(proposal, observation.device.platform)

    def _check_tap_repetition(
        self,
        proposal: ActionProposal,
        source: ActionSource,
        repetition: TapRepetition | None,
    ) -> ActionRefusal | None:
        if repetition is None:
            return None
        if proposal.action is not ActionKind.TAP:
            return self._refuse(
                proposal,
                ActionRefusalCode.INVALID_CONSTRAINT,
                "Only a tap action can contain repetition.",
            )
        if source is not ActionSource.DETERMINISTIC_TEST:
            return self._refuse(
                proposal,
                ActionRefusalCode.ACTION_NOT_ALLOWED,
                "Only a deterministic test can repeat a tap.",
            )
        return None

    def _check_keyboard_and_back(
        self,
        proposal: ActionProposal,
        observation: Observation,
        expected_back_destination_observation_id: str | None,
    ) -> ActionRefusal | None:
        definition = ACTION_DEFINITIONS[proposal.action]
        if observation.keyboard_visible is True and definition.blocked_when_keyboard_visible:
            return self._refuse(
                proposal,
                ActionRefusalCode.KEYBOARD_RULE,
                "Dismiss the visible keyboard before pointer or navigation actions.",
            )
        requires_existing_focus = definition.requires_keyboard_visible and not (
            proposal.action is ActionKind.TYPE and proposal.target_expression is not None
        )
        if requires_existing_focus and observation.keyboard_visible is not True:
            return self._refuse(
                proposal,
                ActionRefusalCode.KEYBOARD_RULE,
                "Focus an editable field before using this action.",
            )
        if proposal.action is ActionKind.BACK:
            if observation.device.platform is not DevicePlatform.ANDROID:
                return self._refuse(
                    proposal,
                    ActionRefusalCode.UNSUPPORTED_ACTION,
                    "Back navigation is available only on Android.",
                )
            if (
                expected_back_destination_observation_id is None
                and not self._limits.allow_exploratory_back
            ):
                return self._refuse(
                    proposal,
                    ActionRefusalCode.INVALID_CONSTRAINT,
                    "Back navigation requires a destination guard.",
                )
        if (
            proposal.action is ActionKind.DISMISS_KEYBOARD
            and observation.keyboard_visible is not True
        ):
            return self._refuse(
                proposal,
                ActionRefusalCode.KEYBOARD_RULE,
                "Keyboard dismissal requires a visible keyboard.",
            )
        return None

    def _check_duration(
        self,
        proposal: ActionProposal,
        platform: DevicePlatform,
    ) -> ActionRefusal | None:
        duration = proposal.duration_seconds
        if proposal.action is ActionKind.WAIT and (
            duration is None
            or duration < MINIMUM_WAIT_SECONDS
            or duration > self._limits.maximum_wait_seconds
        ):
            return self._refuse(
                proposal, ActionRefusalCode.INVALID_CONSTRAINT, "Wait duration exceeds policy."
            )
        if proposal.action is ActionKind.LONG_PRESS and (
            duration is None
            or duration < MINIMUM_LONG_PRESS_SECONDS
            or duration > self._limits.maximum_gesture_seconds
        ):
            return self._refuse(
                proposal,
                ActionRefusalCode.INVALID_CONSTRAINT,
                "Long-press duration exceeds policy.",
            )
        minimum_drag = (
            MINIMUM_IOS_DRAG_SECONDS
            if platform is DevicePlatform.IOS
            else MINIMUM_ANDROID_DRAG_SECONDS
        )
        if proposal.action is ActionKind.DRAG and (
            duration is None
            or duration < minimum_drag
            or duration > self._limits.maximum_gesture_seconds
        ):
            return self._refuse(
                proposal, ActionRefusalCode.INVALID_CONSTRAINT, "Drag duration exceeds policy."
            )
        return None

    def _resolve_grounding(
        self,
        proposal: ActionProposal,
        observation: Observation,
        expected_expression: str | None,
        expected_operation: GroundingTargetOperation | None,
        evidence: GroundingEvidence | None,
    ) -> tuple[PixelPoint | None, PixelRegion | None, ActionRefusal | None]:
        if expected_expression is None:
            return None, None, None
        if evidence is None or expected_operation is None:
            return (
                None,
                None,
                self._refuse(
                    proposal, ActionRefusalCode.INVALID_GROUNDING, "Required grounding is missing."
                ),
            )
        request = evidence.request
        result = evidence.result
        if request.expression != expected_expression or request.operation is not expected_operation:
            return (
                None,
                None,
                self._refuse(
                    proposal,
                    ActionRefusalCode.INVALID_GROUNDING,
                    "Grounding does not match the proposal target.",
                ),
            )
        screenshot = observation.redacted_screenshot
        if (
            request.screenshot.artifact_id != screenshot.artifact_id
            or request.screenshot.sha256 != screenshot.sha256
            or result.source_screenshot_sha256 != screenshot.sha256
        ):
            return (
                None,
                None,
                self._refuse(
                    proposal, ActionRefusalCode.STALE_GROUNDING, "Grounding screenshot is stale."
                ),
            )
        if result.disposition is not GroundingDisposition.ACCEPTED:
            return (
                None,
                None,
                self._refuse(
                    proposal, ActionRefusalCode.INVALID_GROUNDING, "Grounding was not accepted."
                ),
            )
        location = result.location
        if isinstance(location, GroundedPoint):
            if expected_operation is GroundingTargetOperation.SCROLL_REGION:
                return (
                    None,
                    None,
                    self._refuse(
                        proposal,
                        ActionRefusalCode.INVALID_GROUNDING,
                        "A targeted scroll requires a grounded region.",
                    ),
                )
            point = location.pixel
            if not self._point_in_screenshot(point, observation) or self._point_in_system_region(
                point, observation.system_regions
            ):
                return (
                    None,
                    None,
                    self._refuse(
                        proposal,
                        ActionRefusalCode.INVALID_GROUNDING,
                        "Grounded point is outside the application-owned screen.",
                    ),
                )
            return point, None, None
        if isinstance(location, GroundedRegion) and expected_operation is (
            GroundingTargetOperation.SCROLL_REGION
        ):
            region = location.pixel
            if not self._region_in_screenshot(region, observation) or self._overlaps_system_region(
                region, observation.system_regions
            ):
                return (
                    None,
                    None,
                    self._refuse(
                        proposal,
                        ActionRefusalCode.INVALID_GROUNDING,
                        "Grounded region is outside the application-owned screen.",
                    ),
                )
            return None, region, None
        return (
            None,
            None,
            self._refuse(
                proposal,
                ActionRefusalCode.INVALID_GROUNDING,
                "Grounding shape does not match the action.",
            ),
        )

    @staticmethod
    def _point_in_screenshot(point: PixelPoint, observation: Observation) -> bool:
        screenshot = observation.redacted_screenshot
        return point.x < screenshot.width_px and point.y < screenshot.height_px

    @staticmethod
    def _region_in_screenshot(region: PixelRegion, observation: Observation) -> bool:
        screenshot = observation.redacted_screenshot
        return region.right <= screenshot.width_px and region.bottom <= screenshot.height_px

    @staticmethod
    def _point_in_system_region(
        point: PixelPoint, regions: tuple[ScreenRegion, ...] | None
    ) -> bool:
        return any(
            region.left <= point.x < region.right and region.top <= point.y < region.bottom
            for region in regions or ()
        )

    @staticmethod
    def _overlaps_system_region(
        target: PixelRegion, regions: tuple[ScreenRegion, ...] | None
    ) -> bool:
        return any(
            target.left < region.right
            and target.right > region.left
            and target.top < region.bottom
            and target.bottom > region.top
            for region in regions or ()
        )

    @staticmethod
    def _refuse(proposal: ActionProposal, code: ActionRefusalCode, reason: str) -> ActionRefusal:
        return ActionRefusal(
            proposal_id=proposal.proposal_id,
            code=code,
            reason=reason,
            sensitivity=proposal.sensitivity,
            retention=proposal.retention,
        )
