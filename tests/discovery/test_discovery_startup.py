"""Cold discovery compiles the exact knowledge used by generation and execution."""

from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery import coordinator as coordinator_module
from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.contracts import (
    CapabilityCandidate,
    DiscoveryBlocker,
    JourneyGoal,
    CapabilityState,
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTerminationSignals,
    DiscoveryTransitionEvidence,
    ObservedControl,
    SideEffectClass,
    finalize_discovery,
)
from yewo.discovery.runtime import DiscoveryResult
from yewo.discovery.map_builder import build_app_map
from yewo.discovery.store import DiscoveryStore
from yewo.testing.discovery_startup import DiscoveryStartupKnowledgeProvider
from yewo.testing.models import UserJourney


def _terminal_run(header, budgets, *, cyclic=False) -> tuple[DiscoveryRun, DiscoveryRun]:
    goal = JourneyGoal(statement="reach the editor")
    open_run = DiscoveryRun(
        header=header("discovery-run"),
        goals=(goal,),
        observations=(
            DiscoveryObservation(
                observation_id="observation-list",
                screen_state_id="state-list",
                phash="0" * 64,
                controls=(
                    ObservedControl(
                        control_identity="control-edit",
                        role="button",
                        bounds=RelativeBounds(left=0, top=0, right=100, bottom=100),
                        actionable=True,
                    ),
                ),
            ),
            DiscoveryObservation(
                observation_id="observation-editor",
                screen_state_id="state-editor",
                phash="f" * 64,
                controls=(
                    ObservedControl(
                        control_identity="control-list",
                        role="button",
                        bounds=RelativeBounds(left=0, top=0, right=100, bottom=100),
                        actionable=True,
                    ),
                ),
            ),
        ),
        transitions=(
            DiscoveryTransitionEvidence(
                transition_identity="transition-editor",
                source_observation_id="observation-list",
                destination_observation_id="observation-editor",
                control_identity="control-edit",
                action_kind="tap",
                authorization_receipt_id="authorization-editor",
                dispatch_receipt_id="dispatch-editor",
                settlement_receipt_id="settlement-editor",
                side_effect=SideEffectClass.READ_ONLY,
                reversible=True,
            ),
            *(
                (
                    DiscoveryTransitionEvidence(
                        transition_identity="transition-list",
                        source_observation_id="observation-editor",
                        destination_observation_id="observation-list",
                        control_identity="control-list",
                        action_kind="tap",
                        authorization_receipt_id="authorization-list",
                        dispatch_receipt_id="dispatch-list",
                        settlement_receipt_id="settlement-list",
                        side_effect=SideEffectClass.READ_ONLY,
                        reversible=True,
                    ),
                )
                if cyclic
                else ()
            ),
        ),
        capability_candidates=(
            CapabilityCandidate(
                capability="editor",
                goal_ids=(goal.goal_id,),
                observation_ids=("observation-editor",),
                confidence=1,
            ),
        ),
        budgets=budgets,
    )
    closed = open_run.model_copy(
        update={
            "goals": (
                goal.activate().close(
                    CapabilityState.ROUTE_VERIFIED,
                    ("observation-editor",),
                ),
            )
        }
    )
    terminal = seal_artifact(
        finalize_discovery(
            closed,
            DiscoveryTerminationSignals(relevant_frontier_remaining=False),
        )
    )
    return open_run, terminal


def test_a_cyclic_map_keeps_the_discovery_root_for_route_compilation(
    tmp_path, header, budgets, compatibility
):
    del compatibility
    _, terminal = _terminal_run(header, budgets, cyclic=True)
    app_map = build_app_map((terminal,))
    store = DiscoveryStore(tmp_path / "knowledge")
    provider = DiscoveryStartupKnowledgeProvider(
        store,
        app_map.header.compatibility,
        lambda: terminal,
        None,
        None,
    )

    route_memory = provider._compile_route(
        app_map,
        terminal.goals[0].statement,
        terminal.goals[0].evidence_observation_ids,
        (),
    )

    assert len(app_map.root_node_ids) == 1
    assert route_memory is not None
    assert route_memory.start.node_id == app_map.root_node_ids[0]


def test_cold_discovery_compiles_and_persists_one_startup_knowledge_set(
    monkeypatch,
    tmp_path,
    header,
    budgets,
    compatibility,
):
    open_run, terminal = _terminal_run(header, budgets)
    received_scenario_values: list[tuple[str, ...]] = []

    class Controller:
        def __init__(self, received, *, scenario_owned_values=()):
            assert received is open_run
            received_scenario_values.append(tuple(scenario_owned_values))

        def run_to_terminal(self, runtime, proposer, *, interrupted):
            del runtime, proposer
            assert not interrupted()
            return DiscoveryResult(run=terminal, findings=())

    monkeypatch.setattr(coordinator_module, "DiscoveryController", Controller)
    store = DiscoveryStore(tmp_path / "knowledge")
    provider = DiscoveryStartupKnowledgeProvider(
        store,
        compatibility,
        lambda: open_run,
        object(),
        object(),
    )

    knowledge = provider.discover(
        UserJourney(
            id="journey",
            name="Journey",
            description="Reach the editor.",
            values={"card": "4242424242424242"},
        )
    )

    # Without this the leak guard is constructed but never given anything to
    # refuse, so a declared value could reach the persisted run and the map.
    assert received_scenario_values == [("4242424242424242",)]
    assert knowledge is not None
    assert len(knowledge.generation_evidence.capabilities) == 1
    assert len(knowledge.generation_evidence.route_prefixes) == 1
    assert knowledge.route_preparer is not None
    assert len(tuple((tmp_path / "knowledge" / "discovery_runs").glob("*.json"))) == 1
    assert len(tuple((tmp_path / "knowledge" / "app_maps").glob("*.json"))) == 1
    assert len(tuple((tmp_path / "knowledge" / "route_memories").glob("*.json"))) == 1


def test_second_startup_reuses_sealed_knowledge_and_compiles_nothing_new(
    monkeypatch,
    tmp_path,
    header,
    budgets,
    compatibility,
):
    open_run, terminal = _terminal_run(header, budgets)
    discovery_runs = []

    class Controller:
        def __init__(self, received, *, scenario_owned_values=()):
            del scenario_owned_values
            discovery_runs.append(received)

        def run_to_terminal(self, runtime, proposer, *, interrupted):
            del runtime, proposer, interrupted
            return DiscoveryResult(run=terminal, findings=())

    monkeypatch.setattr(coordinator_module, "DiscoveryController", Controller)
    store = DiscoveryStore(tmp_path / "knowledge")
    provider = DiscoveryStartupKnowledgeProvider(
        store,
        compatibility,
        lambda: open_run,
        object(),
        object(),
    )
    journey = UserJourney(id="journey", name="Journey", description="Reach the editor.")

    first = provider.discover(journey)
    second = provider.discover(journey)

    assert len(discovery_runs) == 1
    assert second is not None and first is not None
    first_evidence = first.generation_evidence
    second_evidence = second.generation_evidence
    assert first_evidence is not None and second_evidence is not None
    # What authorizes generation is identical, which is what a lock records.
    # What each run reported about itself is not: the first discovered and the
    # second reused, so only the first reached anything.
    assert second_evidence.evidence_sha256 == first_evidence.evidence_sha256
    assert second_evidence.capabilities == first_evidence.capabilities
    assert second_evidence.route_prefixes == first_evidence.route_prefixes
    assert second_evidence.reached_capabilities == ()
    assert len(tuple((tmp_path / "knowledge" / "app_maps").glob("*.json"))) == 1
    assert len(tuple((tmp_path / "knowledge" / "route_memories").glob("*.json"))) == 1


def test_a_blocked_run_yields_no_knowledge_instead_of_aborting_the_journey(
    monkeypatch, tmp_path, header, budgets, compatibility
):
    """A run that reached no screen has no map. Building one aborted the journey."""
    unobserved = DiscoveryRun(
        header=header("discovery-run"),
        goals=(JourneyGoal(statement="reach the editor"),),
        budgets=budgets,
    )
    blocked = seal_artifact(
        finalize_discovery(
            unobserved, DiscoveryTerminationSignals(blocker=DiscoveryBlocker.SETUP)
        )
    )

    class Controller:
        def __init__(self, received, *, scenario_owned_values=()):
            del received, scenario_owned_values

        def run_to_terminal(self, runtime, proposer, *, interrupted):
            del runtime, proposer, interrupted
            return DiscoveryResult(run=blocked, findings=())

    monkeypatch.setattr(coordinator_module, "DiscoveryController", Controller)
    provider = DiscoveryStartupKnowledgeProvider(
        DiscoveryStore(tmp_path / "knowledge"),
        compatibility,
        lambda: unobserved,
        object(),
        object(),
    )

    knowledge = provider.discover(
        UserJourney(id="journey", name="Journey", description="Reach the editor.")
    )

    assert blocked.observations == ()
    assert knowledge is None
