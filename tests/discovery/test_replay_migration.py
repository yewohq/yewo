"""Legacy replay is input to revalidation, never silently promoted knowledge."""

from yewo.agent.memory.replay_store import (
    CachedAction,
    CachedTransition,
    ElementAnchor,
    RelativeBounds,
    ReplayTrace,
    ScreenSignature,
    TraceIdentity,
)
from yewo.discovery.replay_migration import (
    ReplayMigrationDisposition,
    assess_replay_migration,
)
from yewo.agent.run.postconditions import EnteredValuePostcondition


def _identity():
    return TraceIdentity(
        scenario_uuid="isolated-scenario-id",
        scenario_digest="scenario-digest",
        platform="android",
        target_profile="phone",
        detector="yolo-ui",
        perception_pipeline="fusion-v1",
        grounding="marks",
    )


def _screen(phash):
    return ScreenSignature(phash=phash, element_count=1)


def _target():
    return ElementAnchor(
        role="button",
        text="safe-keyed-digest",
        bounds=RelativeBounds(left=100, top=100, right=300, bottom=200),
        twins=1,
    )


def _trace(app_digest, action=None):
    return ReplayTrace(
        identity=_identity(),
        learned_from=app_digest,
        transitions=(
            CachedTransition(
                position=1,
                source=_screen("0" * 64),
                action=action or CachedAction(kind="tap", target=_target()),
                destination=_screen("1" * 64),
                arrival_ms=250,
            ),
        ),
    )


def test_safe_legacy_navigation_requires_live_revalidation(compatibility):
    assessment = assess_replay_migration(
        _trace(compatibility.app_artifact_sha256),
        compatibility,
        "d" * 64,
    )

    assert assessment.disposition is ReplayMigrationDisposition.REVALIDATION_REQUIRED
    assert assessment.reusable_transition_positions == (1,)
    assert "isolated-scenario-id" not in assessment.model_dump_json()


def test_trace_for_an_old_app_artifact_is_rejected(compatibility):
    assessment = assess_replay_migration(
        _trace("e" * 64), compatibility, "d" * 64
    )

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED
    assert "different app artifact" in assessment.reasons[0]


def test_value_carrying_replay_is_not_shared(compatibility):
    trace = _trace(
        compatibility.app_artifact_sha256,
        CachedAction(kind="type", target=_target(), value_name="password"),
    )

    assessment = assess_replay_migration(trace, compatibility, "d" * 64)

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED
    assert assessment.reusable_transition_positions == ()


def test_unidentified_app_artifact_is_not_assumed_compatible(compatibility):
    trace = _trace(compatibility.app_artifact_sha256).model_copy(
        update={"learned_from": None}
    )

    assessment = assess_replay_migration(trace, compatibility, "d" * 64)

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED
    assert "does not identify" in assessment.reasons[0]


def test_unknown_legacy_schema_is_rejected(compatibility):
    trace = _trace(compatibility.app_artifact_sha256).model_copy(
        update={"schema_version": 99}
    )

    assessment = assess_replay_migration(trace, compatibility, "d" * 64)

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED
    assert "schema or matcher" in assessment.reasons[0]


def test_tap_without_reacquisition_evidence_is_not_reused(compatibility):
    trace = _trace(
        compatibility.app_artifact_sha256,
        CachedAction(kind="tap", target=None),
    )

    assessment = assess_replay_migration(trace, compatibility, "d" * 64)

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED


def test_tap_with_scenario_postconditions_is_not_reused(compatibility):
    action = CachedAction(
        kind="tap",
        target=_target(),
        postconditions=(EnteredValuePostcondition(value_name="password"),),
    )

    assessment = assess_replay_migration(
        _trace(compatibility.app_artifact_sha256, action),
        compatibility,
        "d" * 64,
    )

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED


def test_transition_without_exact_screen_hashes_is_not_reused(compatibility):
    trace = _trace(compatibility.app_artifact_sha256)
    transition = trace.transitions[0].model_copy(update={"source": ScreenSignature()})
    trace = trace.model_copy(update={"transitions": (transition,)})

    assessment = assess_replay_migration(trace, compatibility, "d" * 64)

    assert assessment.disposition is ReplayMigrationDisposition.REJECTED
