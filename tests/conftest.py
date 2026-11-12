"""Boundaries every test runs inside."""

from collections.abc import Iterator
import socket

import pytest


@pytest.fixture(autouse=True)
def _telemetry_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Turn telemetry off for the suite, which must not reach the network.

    Set here rather than detected in `src`. Production code that recognises a test runner
    behaves differently under test than in front of a user, which leaves the behaviour a user
    gets unexercised.
    """
    monkeypatch.setenv("YEWO_TELEMETRY", "off")
    yield


@pytest.fixture(autouse=True)
def _network_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject accidental external connections from offline tests."""

    def reject_network(*unused_args: object, **unused_kwargs: object) -> None:
        pytest.fail("Offline tests cannot open network connections.")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
