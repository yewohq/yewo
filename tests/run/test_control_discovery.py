"""State-contract tests for deterministic missing-control discovery."""

import pytest

from yewo.agent.run.control_discovery import (
    ControlDiscoveryState,
    ControlDiscoveryTransaction,
)


def _transaction(probe_budget=2):
    return ControlDiscoveryTransaction(
        source_phash="screen",
        region="middle_left",
        description="repetitions field",
        intent="enter_text",
        probe_budget=probe_budget,
    )


def test_success_follows_the_only_legal_path():
    transaction = _transaction()
    transaction.begin_probing()
    transaction.record_probe(0, None)
    transaction.begin_probing()
    transaction.record_probe(1, (10, 20, 30, 40))
    transaction.resolve((11, 21, 31, 41))

    assert transaction.state is ControlDiscoveryState.RESOLVED
    assert transaction.verified_bounds == (11, 21, 31, 41)
    assert transaction.probes_remaining == 0


def test_probe_budget_and_unique_cells_are_enforced():
    transaction = _transaction(probe_budget=1)
    transaction.begin_probing()
    transaction.record_probe(0, None)

    with pytest.raises(ValueError, match="twice"):
        transaction.record_probe(0, None)
    with pytest.raises(ValueError, match="budget"):
        transaction.record_probe(1, None)


@pytest.mark.parametrize("terminal", ["resolved", "failed"])
def test_terminal_state_is_immutable(terminal):
    transaction = _transaction()
    transaction.begin_probing()
    if terminal == "resolved":
        transaction.record_probe(0, (10, 20, 30, 40))
        transaction.resolve((10, 20, 30, 40))
    else:
        transaction.fail("not found")

    with pytest.raises(ValueError):
        transaction.begin_probing()
    with pytest.raises(ValueError):
        transaction.fail("changed")


def test_resolve_requires_fresh_verification():
    transaction = _transaction()

    with pytest.raises(ValueError, match="verifying"):
        transaction.resolve((10, 20, 30, 40))
