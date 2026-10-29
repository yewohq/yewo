"""Per-step action history built from system-observed outcomes."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ActionResult:
    """System-observed outcome of a single executed action."""

    success: bool
    observation: str
    error: Optional[str] = None


@dataclass
class ActionEntry:
    """One step in the action history and its system-observed result."""

    step_num: int
    action_desc: str
    result: ActionResult
    reaction: str = ""    # system-observed screen change description, filled at next step via update_last_reaction()


class AgentHistory:
    """Rolling action history with outcomes, formatted for LLM consumption."""

    def __init__(self):
        self._entries: List[ActionEntry] = []

    def append(self, entry: ActionEntry) -> None:
        self._entries.append(entry)

    def update_last_reaction(self, reaction: str) -> None:
        """Attach the system-observed screen change description to the previous step."""
        if self._entries and reaction:
            self._entries[-1].reaction = reaction

    def mark_last_failed(self, observation: str) -> None:
        """Replace the latest action's provisional device outcome."""
        if self._entries:
            self._entries[-1].result.success = False
            self._entries[-1].result.observation = observation

    def format_for_prompt(self) -> str:
        """Return authoritative observations only; omit model self-assessment."""
        if not self._entries:
            return "No actions executed yet."
        lines = []
        for e in self._entries:
            line = f"[Step {e.step_num}] {e.action_desc} → {e.result.observation}"
            if e.reaction:
                line += f"\n         REACTION: {e.reaction}"
            lines.append(line)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
