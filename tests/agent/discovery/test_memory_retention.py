"""App-memory graph and observation retention tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent import AppMemory
from yewo.agent.discovery import merge_replay_variants
from yewo.run_narration import RunNarrator, narration_scope

from agent.discovery.support import observation
from agent.discovery.test_promotion import _promote


def test_replay_variant_retention_preserves_referenced_transition_observations(
    tmp_path: Path,
) -> None:
    promotion = _promote()
    assert promotion.memory is not None
    compatibility = promotion.memory.compatibility.model_copy(
        update={
            "retention": promotion.memory.compatibility.retention.model_copy(
                update={"maximum_variants_per_screen": 1}
            )
        }
    )
    memory = AppMemory.model_validate(
        {
            **promotion.memory.model_dump(mode="python"),
            "compatibility": compatibility,
        }
    )
    fresh_source = observation("fresh_source", "0000000000000000")
    fresh_destination = observation(
        "fresh_destination",
        "1111111111111111",
        predecessor_observation_id=fresh_source.observation_id,
        offset_seconds=1,
    )
    newest_source = observation("newest_source", "0000000000000000", offset_seconds=2)
    newest_destination = observation(
        "newest_destination",
        "1111111111111111",
        predecessor_observation_id=newest_source.observation_id,
        offset_seconds=3,
    )

    with narration_scope(RunNarrator(tmp_path)):
        updated = merge_replay_variants(
            memory,
            {
                fresh_source.observation_id: fresh_source,
                fresh_destination.observation_id: fresh_destination,
                newest_source.observation_id: newest_source,
                newest_destination.observation_id: newest_destination,
            },
            source_run_id="replay_run",
            updated_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        )

    retained_observation_ids = {
        variant.observation_id for screen in updated.screens for variant in screen.variants
    }
    transition = updated.transitions[0]
    assert transition.source_observation_id in retained_observation_ids
    assert transition.destination_observation_id in retained_observation_ids
    assert fresh_source.observation_id not in retained_observation_ids
    assert fresh_destination.observation_id not in retained_observation_ids
    assert newest_source.observation_id in retained_observation_ids
    assert newest_destination.observation_id in retained_observation_ids
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Memory retention removed 2 older optional_variant record(s)." in narration


def test_referenced_hub_variants_do_not_consume_the_optional_variant_limit() -> None:
    promotion = _promote()
    assert promotion.memory is not None
    memory = promotion.memory
    transition = memory.transitions[0]
    source_screen = next(
        screen for screen in memory.screens if screen.screen_id == transition.source_screen_id
    )
    source_variant = source_screen.variants[0]
    transitions = []
    source_variants = []
    for index in range(23):
        observation_id = f"hub_{index}"
        source_variants.append(
            source_variant.model_copy(update={"observation_id": observation_id})
        )
        grounding = transition.target_grounding
        assert grounding is not None
        transitions.append(
            transition.model_copy(
                update={
                    "transition_id": f"transition_{index}",
                    "source_observation_id": observation_id,
                    "target_grounding": grounding.model_copy(
                        update={"source_observation_id": observation_id}
                    ),
                }
            )
        )
    screens = tuple(
        screen.model_copy(update={"variants": tuple(source_variants)})
        if screen.screen_id == source_screen.screen_id
        else screen
        for screen in memory.screens
    )

    retained = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "screens": screens,
            "transitions": tuple(transitions),
            "checkpoints": (),
            "routes": (),
            "capabilities": (),
        }
    )

    retained_source = next(
        screen for screen in retained.screens if screen.screen_id == source_screen.screen_id
    )
    assert len(retained_source.variants) == 23
