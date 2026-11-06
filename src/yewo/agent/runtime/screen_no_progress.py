"""Count no-progress evidence on equivalent logical screens."""

from __future__ import annotations

from yewo.observations import ScreenComparisonPolicy
from yewo.agent.memory.screens import observations_match
from yewo.observations import Observation


class ScreenNoProgressCounts:
    """Own logical-screen counters independently from refusal policy."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self.reset()

    def reset(self) -> None:
        self._counts: list[tuple[Observation, int]] = []

    def count(self, observation: Observation) -> int:
        index = self._index(observation)
        return 0 if index is None else self._counts[index][1]

    def record(self, observation: Observation) -> int:
        index = self._index(observation)
        if index is None:
            count = 1
            self._counts.append((observation, count))
        else:
            screen, previous = self._counts[index]
            count = previous + 1
            self._counts[index] = (screen, count)
        return count

    def clear(self, observation: Observation) -> None:
        self._counts = [
            (screen, count)
            for screen, count in self._counts
            if not observations_match(observation, screen, self._comparison)
        ]

    def _index(self, observation: Observation) -> int | None:
        return next(
            (
                index
                for index, (screen, unused_count) in enumerate(self._counts)
                if observations_match(observation, screen, self._comparison)
            ),
            None,
        )


__all__ = ["ScreenNoProgressCounts"]
