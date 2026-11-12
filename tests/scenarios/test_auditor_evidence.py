"""Tests for bounded scenario-auditor evidence selection."""

from yewo.agent import AgentTerminalReason
from yewo.agent.planning.evidence import (
    PlannerDecisionMetadata,
    PlannerDecisionSource,
)
from yewo.observations import Observation
from yewo.scenarios.auditor_evidence import (
    _build_visual_episodes,
    _required_observation_labels,
    _visual_frames,
)
from yewo.scenarios.records import (
    DiscoveredTestSurface,
    DiscoveryProvenance,
    ScenarioSurfaceKind,
)

from agent.discovery.support import agent_result, observation


def _replay_step(index: int, before_id: str, after_id: str):
    before = observation(before_id, f"{index:016x}")
    after = observation(
        after_id,
        f"{index + 1:016x}",
        predecessor_observation_id=before_id,
    )
    source = agent_result(
        before,
        after,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    return source.model_copy(
        update={
            "step_id": f"step_{index}",
            "step_index": index + 1,
            "decision": PlannerDecisionMetadata(
                source=PlannerDecisionSource.COMPILED_PROCEDURE,
                reused_flow_id="procedure",
                reused_procedure_step_ids=(f"procedure_step_{index}",),
                reused_segment_index=index,
            ),
        }
    )


def test_consistency_evidence_includes_first_following_observation() -> None:
    surface = DiscoveredTestSurface(
        surface_id="surface_checkout_source",
        journey_id="journey",
        kind=ScenarioSurfaceKind.PRESENTATION,
        summary="The payment selection is visible.",
        observable_outcomes=("The selected payment method is visible.",),
        state_ids=("payment",),
        provenance=DiscoveryProvenance(
            discovery_run_id="discovery_run",
            discovery_objective_id="objective",
            procedure_id="procedure",
            verification_id="verification",
            observation_ids=("observation_payment",),
            procedure_step_indexes=(11,),
        ),
    )
    source = _replay_step(11, "before_payment", "observation_payment")
    following = _replay_step(12, "before_checkout", "observation_checkout")

    labels = _required_observation_labels(
        (surface,),
        (),
        (source, following),
        include_following_observation=True,
    )

    assert labels == {
        "observation_payment": ("The payment selection is visible.",),
        "observation_checkout": ("First settled observation after the required test surfaces.",),
    }


def test_visual_frames_include_model_settlement_recovery_and_final_states() -> None:
    before = observation("before", "0000000000000001")
    settled_observation = observation(
        "settled",
        "0000000000000002",
        predecessor_observation_id="before",
    )
    recovery = observation(
        "recovery",
        "0000000000000003",
        predecessor_observation_id="settled",
    )
    result = agent_result(
        before,
        settled_observation,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    )
    step = result.steps[0].model_copy(
        update={
            "settlement_observation_ids": ("intermediate", "settled"),
            "recovery_observation_id": "recovery",
        }
    )
    result = result.model_copy(
        update={"steps": (step,), "final_observation_id": recovery.observation_id}
    )

    frames, covered_surface_ids = _visual_frames((result,), ())

    assert tuple(frame.observation_id for frame in frames) == (
        "before",
        "intermediate",
        "settled",
        "recovery",
    )
    assert tuple(frame.phase for frame in frames) == (
        "source",
        "settlement",
        "settlement",
        "recovery",
    )
    assert covered_surface_ids == frozenset()


def test_visual_episodes_keep_distinct_transients_and_merge_exact_duplicates() -> None:
    """Settlement may deduplicate a stable pair, but not a transient defect before it."""

    def with_screenshot_digest(item: Observation, digest: str) -> Observation:
        return item.model_copy(
            update={"screenshot": item.screenshot.model_copy(update={"sha256": digest})}
        )

    before = with_screenshot_digest(observation("before", "0000000000000001"), "1" * 64)
    intermediate = with_screenshot_digest(
        observation("intermediate", "0000000000000002"),
        "2" * 64,
    )
    settled = with_screenshot_digest(
        observation("settled", "0000000000000003"),
        "3" * 64,
    )
    duplicate = with_screenshot_digest(
        observation("settled_duplicate", "0000000000000003"),
        "3" * 64,
    )
    result = agent_result(
        before,
        settled,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    )
    step = result.steps[0].model_copy(
        update={
            "settlement_observation_ids": (
                intermediate.observation_id,
                settled.observation_id,
                duplicate.observation_id,
            )
        }
    )
    result = result.model_copy(
        update={"steps": (step,), "final_observation_id": duplicate.observation_id}
    )
    frames, _unused_surface_ids = _visual_frames((result,), ())

    episodes = _build_visual_episodes(
        frames,
        {item.observation_id: item for item in (before, intermediate, settled, duplicate)},
    )

    images = episodes[0].images
    assert tuple(image.observation_id for image in images) == (
        "before",
        "intermediate",
        "settled",
    )
    assert images[-1].related_observation_ids == ("settled_duplicate",)


def test_visual_frames_record_required_surface_coverage_without_dropping_repairs() -> None:
    surface = DiscoveredTestSurface(
        surface_id="surface_sign_in",
        journey_id="journey",
        kind=ScenarioSurfaceKind.PRESENTATION,
        summary="The sign-in screen is visible.",
        observable_outcomes=("The sign-in controls are visible.",),
        state_ids=("sign_in",),
        provenance=DiscoveryProvenance(
            discovery_run_id="discovery_run",
            discovery_objective_id="objective",
            procedure_id="procedure",
            verification_id="verification",
            observation_ids=("discovery_sign_in",),
            procedure_step_indexes=(0,),
        ),
    )
    replay = _replay_step(0, "before_replay", "after_replay")
    repair = _replay_step(1, "before_repair", "after_repair").model_copy(
        update={
            "decision": PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL),
        }
    )
    result = agent_result(
        observation("start", "0000000000000001"),
        observation(
            "finish",
            "0000000000000002",
            predecessor_observation_id="start",
        ),
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).model_copy(update={"steps": (replay, repair), "final_observation_id": "after_repair"})

    frames, covered_surface_ids = _visual_frames((result,), (surface,))

    assert covered_surface_ids == frozenset({"surface_sign_in"})
    assert "before_repair" in {frame.observation_id for frame in frames}
    assert "after_repair" in {frame.observation_id for frame in frames}
