"""Tests for side-effect-free production coordinator composition."""

from yewo.runs import RunCoordinator, build_default_run_coordinator


def test_default_composition_does_not_discover_or_change_devices() -> None:
    coordinator = build_default_run_coordinator({})

    assert isinstance(coordinator, RunCoordinator)
