from yewo.discovery import coordinator as coordinator_module
from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.contracts import (
    JourneyGoal,
    DiscoveryRun,
    DiscoveryTerminationSignals,
    finalize_discovery,
)
from yewo.discovery.coordinator import DiscoveryCoordinator
from yewo.discovery.runtime import DiscoveryResult


def test_coordinator_automatically_discovers_and_persists_when_map_is_absent(
    monkeypatch, header, budgets, compatibility
):
    run = DiscoveryRun(
        header=header("discovery-run"),
        goals=(JourneyGoal(statement="reach the editor"),),
        budgets=budgets,
    )
    terminal = seal_artifact(
        finalize_discovery(
            run,
            DiscoveryTerminationSignals(relevant_frontier_remaining=False),
        )
    )

    received_scenario_values: list[tuple[str, ...]] = []

    class Controller:
        def __init__(self, received, *, scenario_owned_values=()):
            assert received is run
            received_scenario_values.append(tuple(scenario_owned_values))

        def run_to_terminal(self, runtime, proposer, *, interrupted):
            del runtime, proposer
            assert not interrupted()
            return DiscoveryResult(run=terminal, findings=())

    class Repository:
        saved = None

        def find_compatible_app_map(self, requested):
            assert requested == compatibility
            return None

        def save_discovery_run(self, terminal_run):
            self.saved = terminal_run

    monkeypatch.setattr(coordinator_module, "DiscoveryController", Controller)
    repository = Repository()

    result = DiscoveryCoordinator(repository).ensure_knowledge(
        compatibility,
        lambda: run,
        object(),
        object(),
        scenario_owned_values=("4242424242424242",),
    )

    assert received_scenario_values == [("4242424242424242",)]
    assert result.app_map is None
    assert result.discovery.run == terminal
    assert repository.saved == terminal
