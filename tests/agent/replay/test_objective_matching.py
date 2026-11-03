"""LLM objective matching for audited stored procedures."""

from typing import cast

from yewo.agent.discovery import DiscoveryObjective
from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.replay.procedure.objective_matching import (
    OBJECTIVE_MATCHING_SYSTEM_PROMPT,
    ObjectiveMatcher,
)
from yewo.agent.replay.procedure.store import StoredProcedureBundle
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.procedure.preparation import prepare_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.models import ModelRole, TextModelContent

from agent.planning.support import StubPlannerModel
from agent.replay.test_procedure_compiler import _evidence, _response


def _bundle() -> StoredProcedureBundle:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    return StoredProcedureBundle(
        compatibility=MemoryCompatibility(application=evidence.run.application),
        evidence=evidence,
        procedure=procedure,
    )


def _objective(*, condition_count: int = 1) -> DiscoveryObjective:
    return DiscoveryObjective(
        objective_id="current_objective",
        statement="Sign in to the account.",
        completion_conditions=tuple(
            f"The account home is visible {index}." for index in range(condition_count)
        ),
    )


def test_matcher_selects_only_an_llm_equivalent_audited_procedure() -> None:
    bundle = _bundle()
    model = StubPlannerModel(
        {
            "request_id": "objective_match_request",
            "procedure_id": bundle.procedure.procedure_id,
            "relationship": "equivalent",
            "reason": "Both objectives require the same observable signed-in state.",
        }
    )
    matcher = ObjectiveMatcher(
        model,
        "replay-preparer",
        identifier_factory=lambda: "objective_match_request",
    )

    selection = matcher.match(_objective(), (bundle,))

    assert selection is not None
    assert selection.procedure_id == bundle.procedure.procedure_id
    assert selection.stored_objective_id == bundle.evidence.task.objective_id
    assert selection.preparation_model_requests == 1
    assert matcher.model_request_count == 1
    request = model.requests[0]
    assert request.role is ModelRole.REPLAY_PREPARER
    assert request.messages[0].content == (
        TextModelContent(text=OBJECTIVE_MATCHING_SYSTEM_PROMPT),
    )


def test_matcher_rejects_a_non_equivalent_relationship() -> None:
    bundle = _bundle()
    model = StubPlannerModel(
        {
            "request_id": "objective_match_request",
            "procedure_id": None,
            "relationship": "current_is_broader",
            "reason": "The requested objective requires an additional outcome.",
        }
    )
    matcher = ObjectiveMatcher(
        model,
        "replay-preparer",
        identifier_factory=lambda: "objective_match_request",
    )

    assert matcher.match(_objective(), (bundle,)) is None


def test_matcher_rejects_unknown_procedure_and_condition_count_changes() -> None:
    bundle = _bundle()
    unknown_model = StubPlannerModel(
        {
            "request_id": "objective_match_request",
            "procedure_id": "unknown_procedure",
            "relationship": "equivalent",
            "reason": "The candidate is equivalent.",
        }
    )
    unknown_matcher = ObjectiveMatcher(
        unknown_model,
        "replay-preparer",
        identifier_factory=lambda: "objective_match_request",
    )
    count_model = StubPlannerModel(
        {
            "request_id": "objective_match_request",
            "procedure_id": bundle.procedure.procedure_id,
            "relationship": "equivalent",
            "reason": "The candidate is equivalent.",
        }
    )
    count_matcher = ObjectiveMatcher(
        count_model,
        "replay-preparer",
        identifier_factory=lambda: "objective_match_request",
    )

    assert unknown_matcher.match(_objective(), (bundle,)) is None
    assert count_matcher.match(_objective(condition_count=2), (bundle,)) is None


def test_matcher_makes_no_call_without_a_candidate() -> None:
    model = StubPlannerModel({})
    matcher = ObjectiveMatcher(model, "replay-preparer")

    assert matcher.match(_objective(), ()) is None
    assert matcher.model_request_count == 0
    assert model.requests == []


def test_changed_completion_conditions_are_not_an_exact_procedure_match() -> None:
    bundle = _bundle()

    class StoredProcedure:
        def load(self, compatibility, objective_id, objective):
            del compatibility, objective_id, objective
            return bundle

        def compatible_candidates(self, compatibility):
            del compatibility
            return (bundle,)

    objective = DiscoveryObjective(
        objective_id=bundle.evidence.task.objective_id,
        statement=bundle.evidence.task.objective,
        completion_conditions=("A different observable result is present.",),
    )

    preparation = prepare_replay_procedure(
        objective,
        bundle.compatibility,
        None,
        cast(CompiledProcedureStore, StoredProcedure()),
        lambda: (_ for _ in ()).throw(AssertionError("matcher must respect the budget")),
        "replay-preparer",
        1,
    )

    assert preparation.selection is None
    assert preparation.model_request_count == 0
