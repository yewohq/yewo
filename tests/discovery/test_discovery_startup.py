"""Cold discovery compiles the exact knowledge used by generation and execution."""

from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery import coordinator as coordinator_module
from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.contracts import (
    CapabilityCandidate,
    CapabilityQuestion,
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
from yewo.discovery.store import DiscoveryStore
from yewo.testing.discovery_startup import DiscoveryStartupKnowledgeProvider
from yewo.testing.models import UserJourney


def _terminal_run(header, budgets) -> tuple[DiscoveryRun, DiscoveryRun]:
    question = CapabilityQuestion(intent="reach the editor")
    open_run = DiscoveryRun(
        header=header("discovery-run"),
        questions=(question,),
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
        ),
        capability_candidates=(
            CapabilityCandidate(
                capability="editor",
                question_ids=(question.question_id,),
                observation_ids=("observation-editor",),
                confidence=1,
            ),
        ),
        budgets=budgets,
    )
    closed = open_run.model_copy(
        update={
            "questions": (
                question.activate().close(
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


def test_cold_discovery_compiles_and_persists_one_startup_knowledge_set(
    monkeypatch,
    tmp_path,
    header,
    budgets,
    compatibility,
):
    open_run, terminal = _terminal_run(header, budgets)

    class Controller:
        def __init__(self, received):
            assert received is open_run

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
        UserJourney(id="journey", name="Journey", description="Reach the editor.")
    )

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
        def __init__(self, received):
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
    assert second.generation_evidence == first.generation_evidence
    assert len(tuple((tmp_path / "knowledge" / "app_maps").glob("*.json"))) == 1
    assert len(tuple((tmp_path / "knowledge" / "route_memories").glob("*.json"))) == 1
