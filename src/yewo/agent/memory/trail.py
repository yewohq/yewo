"""Bounded planner summary of screens observed in the current scenario."""

from typing import Optional

from yewo.agent.memory.screen_identity import find_matching_screen


class SessionTrail:
    """Count visual screens without retaining app text, targets, or actions."""

    def __init__(self):
        self._observation_counts: dict[str, int] = {}

    def record_screen(self, phash_hex: str, _raw_elements=None) -> None:
        """Count one visual observation; element content is intentionally ignored."""
        canonical = find_matching_screen(phash_hex, self._observation_counts)
        key = canonical or phash_hex
        self._observation_counts[key] = self._observation_counts.get(key, 0) + 1

    def flush_transition(self, _new_phash: str) -> None:
        """Retain the lifecycle hook without storing historical navigation data."""

    def format_for_prompt(self) -> Optional[str]:
        if not self._observation_counts:
            return None
        observation_count = sum(self._observation_counts.values())
        return (
            "Historical session record only: "
            f"{len(self._observation_counts)} distinct visual screen fingerprint(s) "
            f"across {observation_count} observation(s). Historical target numbers, "
            "coordinates, text, and action descriptions are not retained. Use only "
            "the current screenshot and its current targets for the next action."
        )
