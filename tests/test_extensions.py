"""Extension registration and real-run resource ownership."""

from __future__ import annotations

import pytest

from yewo.extensions import ExtensionRegistry


class _Resource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        start_error: Exception | None = None,
    ) -> None:
        self._name = name
        self._events = events
        self._start_error = start_error

    def start(self) -> None:
        self._events.append(f"{self._name}.start")
        if self._start_error is not None:
            raise self._start_error

    def close(self) -> None:
        self._events.append(f"{self._name}.close")


def test_run_resources_do_not_start_when_registered() -> None:
    events: list[str] = []
    registry = ExtensionRegistry()

    registry.add_run_resource(_Resource("masker", events))

    assert events == []


def test_run_resources_start_once_and_close_in_reverse_order() -> None:
    events: list[str] = []
    registry = ExtensionRegistry()
    registry.add_run_resource(_Resource("first", events))
    registry.add_run_resource(_Resource("second", events))

    registry.start_run_resources()
    registry.start_run_resources()
    registry.close_run_resources()

    assert events == ["first.start", "second.start", "second.close", "first.close"]


def test_start_failure_closes_resources_that_started() -> None:
    events: list[str] = []
    registry = ExtensionRegistry()
    registry.add_run_resource(_Resource("ready", events))
    registry.add_run_resource(
        _Resource("failed", events, start_error=RuntimeError("could not start"))
    )

    with pytest.raises(RuntimeError, match="could not start"):
        registry.start_run_resources()

    assert events == ["ready.start", "failed.start", "failed.close", "ready.close"]
