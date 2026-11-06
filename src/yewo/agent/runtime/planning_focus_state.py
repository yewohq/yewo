"""Own the one-shot planning-focus lifecycle for logical screens."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.memory.screens import observations_match
from yewo.agent.planning.records import (
    PlanningFocusCause,
    PlanningFocusContext,
    ProgressRouteAttempt,
)
from yewo.observations import Observation


class PlanningFocusState:
    """Track when one logical screen earns and consumes focused replanning."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self.reset()

    def reset(self) -> None:
        """Discard focus state at the beginning of a run."""
        self._pending: tuple[Observation, PlanningFocusCause, str] | None = None
        self._focused_screens: list[Observation] = []
        self._action_screen: Observation | None = None

    def request(
        self,
        observation: Observation,
        cause: PlanningFocusCause,
        reason: str,
    ) -> None:
        """Request one focus decision unless this screen already received one."""
        if self._already_focused(observation):
            return
        if self._pending is not None and self._matches(observation, self._pending[0]):
            return
        self._pending = (observation, cause, reason)

    def pending_context(
        self,
        observation: Observation,
        identifier_factory: Callable[[], str],
        unresolved_work: tuple[str, ...],
        prohibited_interaction_intents: tuple[str, ...],
        recent_routes: tuple[ProgressRouteAttempt, ...],
    ) -> PlanningFocusContext | None:
        """Return the focus request owed to the current logical screen."""
        pending = self._pending
        if pending is None or not self._matches(observation, pending[0]):
            return None
        if self._already_focused(observation):
            return None
        unused_screen, cause, reason = pending
        return PlanningFocusContext(
            focus_id=identifier_factory(),
            cause=cause,
            reason=reason,
            unresolved_work=unresolved_work,
            prohibited_interaction_intents=prohibited_interaction_intents,
            recent_routes=recent_routes[-12:],
        )

    def commit(self, observation: Observation) -> None:
        """Claim one pending focus request immediately before planner delivery."""
        self._pending = None
        if self._already_focused(observation):
            return
        self._focused_screens.append(observation)
        self._action_screen = observation

    def action_armed(self, observation: Observation) -> bool:
        """Return whether the next proposal is the one focus-mode exemption."""
        return self._action_screen is not None and self._matches(
            observation,
            self._action_screen,
        )

    def consume_action(self, observation: Observation) -> bool:
        """Consume the focus-mode exemption after one dispatched outcome."""
        if not self.action_armed(observation):
            return False
        self._action_screen = None
        return True

    def _already_focused(self, observation: Observation) -> bool:
        return any(self._matches(observation, prior) for prior in self._focused_screens)

    def _matches(self, first: Observation, second: Observation) -> bool:
        return observations_match(first, second, self._comparison)
