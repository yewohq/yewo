"""One final authorization boundary for every device action source."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from yewo.agent.actuation.actions import (
    Action,
    ActionType,
    KeyEventAction,
    PlannerTapAction,
    ShowKeyboardTargetsAction,
)
from yewo.agent.actuation.orchestrator import ActionOrchestrator
from yewo.agent.actuation.target_not_listed import TargetNotListedAction
from yewo.agent.memory.control_identity import ControlIdentity, build_control_identity
from yewo.agent.run.instruction_controller import InstructionController
from yewo.utils.secrets import SecretScrubber
from yewo.identity import new_uuid


class ActionSource(str, Enum):
    PLANNER = "planner"
    RECOVERY = "recovery"
    SAVED_ROUTE = "saved_route"
    REPLAY = "replay"
    DISCOVERY = "discovery"


REPLAYABLE_ACTION_TYPES = frozenset(
    {ActionType.TAP, ActionType.TYPE}
)
REPLAYABLE_KEYCODES: frozenset[int] = frozenset()


@dataclass(frozen=True)
class ActionAuthorization:
    """The immutable decision immediately preceding one device dispatch."""

    authorization_id: str
    action: Action | PlannerTapAction
    source: ActionSource
    observation_id: str
    control_identity: ControlIdentity
    context_id: str | None
    instruction_id: str | None
    attempt_id: str | None
    expected_destination_phash: str | None = None


class ActionSafetyPolicy:
    """Validate common action invariants regardless of proposal source."""

    def authorize(
        self,
        action: Action | PlannerTapAction,
        *,
        source: ActionSource,
        observation_id: str,
        screen_phash: str | None,
        screen_size: tuple[int, int] | None,
        context_id: str | None = None,
        instruction_controller: InstructionController | None = None,
        keyboard_visible: bool | None = None,
        expected_destination_phash: str | None = None,
    ) -> ActionAuthorization:
        if not observation_id:
            raise ValueError("An action requires an exact source observation.")
        if isinstance(action, ShowKeyboardTargetsAction | TargetNotListedAction):
            raise ValueError("A non-device planner action cannot be dispatched.")
        if source is ActionSource.REPLAY and not self.is_replayable(action):
            raise ValueError("Replay cannot dispatch this action kind.")
        if isinstance(action, KeyEventAction) and action.key == 4 and keyboard_visible:
            raise ValueError(
                "Android Back cannot dismiss the keyboard; use the platform input adapter."
            )
        if (
            isinstance(action, KeyEventAction)
            and action.key == 4
            and expected_destination_phash is None
        ):
            raise ValueError(
                "Android Back requires a previously observed destination postcondition."
            )

        attempt = (
            instruction_controller.begin_attempt(observation_id)
            if instruction_controller is not None
            else None
        )
        return ActionAuthorization(
            authorization_id=new_uuid(),
            action=action,
            source=source,
            observation_id=observation_id,
            control_identity=build_control_identity(action, screen_phash, screen_size),
            context_id=context_id,
            instruction_id=attempt.instruction_id if attempt is not None else None,
            attempt_id=attempt.attempt_id if attempt is not None else None,
            expected_destination_phash=expected_destination_phash,
        )

    @staticmethod
    def is_replayable(action: Action | PlannerTapAction) -> bool:
        if action.type not in REPLAYABLE_ACTION_TYPES:
            return False
        if isinstance(action, KeyEventAction):
            return action.key in REPLAYABLE_KEYCODES
        return True


def execute_authorized(
    authorization: ActionAuthorization,
    device_control,
    scrubber: SecretScrubber | None = None,
):
    """Dispatch only an action carrying a safety-policy decision."""
    return ActionOrchestrator(device_control, scrubber).execute(
        cast(Action, authorization.action)
    )
