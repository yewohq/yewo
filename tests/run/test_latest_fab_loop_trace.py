"""Regression for the 2026-10-28 trainer-app FAB/snackbar loop.

The hashes and coordinates are taken from run
96dbd562-a3a5-4494-b361-0e5b624b7cee.  The old detector split the delivered
and stale events, allowing seventeen selections before the scenario step limit.
"""

from yewo.agent.actuation.actions import Coordinates
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory.loop_detector import RollingLoopDetector

BASE = "daa82d505aa82d5058a225545822b04d2d57f2af2d57f2af2d57d2af2d57d2ab"
SNACKBAR = "d8a02d5058a025504840f2a92d56d2ab2d57d2ab2d57f22fbc77a55fdaafa555"
MENU = "f0a02d5058a025504840f2a82d56d2ab2d57d2ab2d57f22fbc77a55fd00fa555"
SIZE = (1080, 2400)


def _tap(x: int, y: int, target: str) -> TapCoordinateAction:
    return TapCoordinateAction(
        coordinates=Coordinates(x=x, y=y), target=target, reason="continue"
    )


def _delivered_transient(
    detector: RollingLoopDetector, action: TapCoordinateAction
) -> None:
    detector.record_action(action, BASE, SIZE)
    detector.record_delivery_outcome(action, BASE, SNACKBAR, SIZE)
    detector.record_screen(BASE)


def test_real_delivered_and_stale_sequence_quarantines_one_control() -> None:
    detector = RollingLoopDetector()
    fab = _tap(961, 2219, "create a schedule")

    _delivered_transient(detector, fab)
    detector.record_stale(fab, BASE, SIZE)
    message = detector.record_action(fab, BASE, SIZE)

    assert "quarantined" in (message or "")
    assert not detector.action_blocked  # The third attempt may still prove progress.

    detector.record_delivery_outcome(fab, BASE, SNACKBAR, SIZE)
    detector.record_screen(BASE)
    assert "not sent" in (detector.record_stale(fab, BASE, SIZE) or "")
    detector.record_action(fab, BASE, SIZE)

    assert detector.action_blocked
    assert detector.action_repetition_terminated
    assert "quarantined control" in (detector.termination_reason or "")


def test_exploring_an_overlay_does_not_forgive_the_dead_fab() -> None:
    detector = RollingLoopDetector()
    fab = _tap(961, 2219, "create")
    menu = _tap(1021, 204, "overflow")

    for _ in range(3):
        _delivered_transient(detector, fab)

    detector.record_action(menu, BASE, SIZE)
    detector.record_delivery_outcome(menu, BASE, MENU, SIZE)
    detector.record_screen(MENU)
    detector.record_screen(BASE)

    assert "not sent" in (detector.record_action(fab, BASE, SIZE) or "")
    assert detector.action_blocked


def test_an_invented_label_cannot_bypass_coordinate_quarantine() -> None:
    detector = RollingLoopDetector()
    marked = _tap(961, 2219, "Add schedule")
    invented = _tap(963, 2221, "Continue")

    for _ in range(3):
        _delivered_transient(detector, marked)

    assert "not sent" in (detector.record_action(invented, BASE, SIZE) or "")
    assert detector.action_blocked
