"""Detect repeated closed workflows from canonical route evidence."""

from __future__ import annotations

from yewo.agent.actuation import ActionProposal
from yewo.agent.planning.records import ProgressRouteAttempt
from yewo.agent.runtime.interaction_intent import interaction_intent_key
from yewo.observations import Observation


class WorkflowRouteExhaustion:
    """Own route-cycle counts and the interaction intents they exhaust."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._attempts: list[ProgressRouteAttempt] = []
        self._cycle_counts: dict[tuple[str, tuple[str, ...]], int] = {}
        self._exhausted_intents: set[tuple[str, str | None, str, str]] = set()

    def prohibits(
        self,
        proposal: ActionProposal,
        observation: Observation,
        work_item: str,
    ) -> bool:
        key = (
            work_item,
            observation.platform_context,
            observation.screen_fingerprint.value,
            interaction_intent_key(proposal),
        )
        return key in self._exhausted_intents

    def prohibited_intents(
        self,
        observation: Observation,
        work_items: tuple[str, ...],
    ) -> tuple[str, ...]:
        relevant_work = frozenset(work_items)
        return tuple(
            intent
            for work_item, context, fingerprint, intent in self._exhausted_intents
            if work_item in relevant_work
            and context == observation.platform_context
            and fingerprint == observation.screen_fingerprint.value
        )

    def record(self, route: ProgressRouteAttempt, *, task_advanced: bool) -> bool:
        """Return whether this route completed the second identical closed workflow."""
        if task_advanced:
            work_item = route.work_item or ""
            self._attempts = [
                attempt
                for attempt in self._attempts
                if (attempt.work_item or "") != work_item
            ]
            self._cycle_counts = {
                key: count for key, count in self._cycle_counts.items() if key[0] != work_item
            }
            self._exhausted_intents = {
                key for key in self._exhausted_intents if key[0] != work_item
            }
            return False
        self._attempts.append(route)
        current_index = len(self._attempts) - 1
        start_index = next(
            (
                index
                for index in range(current_index - 1, -1, -1)
                if self._attempts[index].work_item == route.work_item
                and self._attempts[index].source_context == route.destination_context
                and self._attempts[index].source_screen_fingerprint
                == route.destination_screen_fingerprint
            ),
            None,
        )
        if start_index is None:
            return False
        cycle = self._attempts[start_index : current_index + 1]
        intents = tuple(
            item.interaction_intent_key
            for item in cycle
            if item.interaction_intent_key is not None
        )
        if not intents:
            return False
        signature = (route.work_item or "", intents)
        count = self._cycle_counts.get(signature, 0) + 1
        self._cycle_counts[signature] = count
        if count < 2:
            return False
        for item in cycle:
            if item.interaction_intent_key is not None:
                self._exhausted_intents.add(
                    (
                        item.work_item or "",
                        item.source_context,
                        item.source_screen_fingerprint or "",
                        item.interaction_intent_key,
                    )
                )
        return True


__all__ = ["WorkflowRouteExhaustion"]
