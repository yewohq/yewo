"""One shared Device Agent budget for a generated scenario."""

from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentUsage,
    remaining_mode_policy,
)


class ScenarioBudgetExhausted(RuntimeError):
    """A scenario route consumed all capacity before remaining work ran."""


class ScenarioBudgetBank:
    """Subtract route and task usage from one scenario-owned policy."""

    def __init__(self, policy: AgentModePolicy) -> None:
        self._remaining: AgentModePolicy | None = policy

    def require_policy(self) -> AgentModePolicy:
        """Return remaining capacity or report scenario budget exhaustion."""
        if self._remaining is None:
            raise ScenarioBudgetExhausted("The scenario Device Agent budget is exhausted.")
        return self._remaining

    def record(self, usage: AgentUsage, elapsed_seconds: float) -> None:
        """Subtract one Device Agent invocation from this scenario bank."""
        self._remaining = remaining_mode_policy(
            self.require_policy(),
            usage,
            elapsed_seconds,
        )
