"""Tests for verified scenario application preconditions."""

from yewo.devices.application_preconditions import ApplicationPreconditionCoordinator


class _Application:
    def __init__(self, *, foreground: bool = True, fails: bool = False) -> None:
        self.foreground = foreground
        self.fails = fails
        self.events: list[str] = []

    def clear_application_data(self) -> None:
        self.events.append("clear")
        self.foreground = False

    def activate(self) -> None:
        self.events.append("activate")
        self.foreground = True

    def relaunch(self) -> None:
        self.events.append("relaunch")
        if self.fails:
            raise RuntimeError("private driver failure")
        self.foreground = True

    def is_foreground(self) -> bool:
        return self.foreground


def _coordinator(application: _Application) -> ApplicationPreconditionCoordinator:
    return ApplicationPreconditionCoordinator(
        application,
        identifier_factory=lambda: "application_setup",
    )


def test_clear_data_reactivates_and_verifies_the_application() -> None:
    application = _Application()

    receipt = _coordinator(application).establish("clear_data")

    assert receipt.verified
    assert receipt.requested_state == "clear_data"
    assert application.events == ["clear", "activate"]


def test_restart_uses_the_verified_relaunch_boundary() -> None:
    application = _Application(foreground=False)

    receipt = _coordinator(application).establish("restart")

    assert receipt.verified
    assert application.events == ["relaunch"]


def test_preserve_is_read_only_and_requires_foreground_state() -> None:
    application = _Application(foreground=False)

    receipt = _coordinator(application).establish("preserve")

    assert not receipt.verified
    assert application.events == []


def test_driver_failure_returns_a_safe_unverified_receipt() -> None:
    application = _Application(fails=True)

    receipt = _coordinator(application).establish("restart")

    assert not receipt.verified
    assert "private" not in receipt.reason
