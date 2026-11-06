"""Track the composition of consecutive pre-dispatch refusals."""

from dataclasses import dataclass


@dataclass
class RefusalStreak:
    """Count refusals and retain whether every one was transient."""

    count: int = 0
    entirely_transient: bool | None = None

    def record(self, is_transient: bool) -> None:
        """Add one refusal to the current streak."""
        if self.entirely_transient is None:
            self.entirely_transient = is_transient
        else:
            self.entirely_transient &= is_transient
        self.count += 1

    def reset(self) -> None:
        """Start a new streak after a dispatched action or a new run."""
        self.count = 0
        self.entirely_transient = None
