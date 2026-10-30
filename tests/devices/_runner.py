"""Deterministic command runner for device inventory tests."""

from __future__ import annotations

from collections.abc import Sequence
from collections import deque
import subprocess


class CommandRunnerStub:
    """Return one configured subprocess result and record the command."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        responses: Sequence[tuple[str, str, int]] | None = None,
    ) -> None:
        configured = responses or ((stdout, stderr, returncode),)
        self._results = deque(
            subprocess.CompletedProcess(
                args=(),
                returncode=response_code,
                stdout=response_stdout,
                stderr=response_stderr,
            )
            for response_stdout, response_stderr, response_code in configured
        )
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self, command: Sequence[str], *, timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((tuple(command), timeout_seconds))
        result = self._results[0]
        if len(self._results) > 1:
            result = self._results.popleft()
        return result
