"""Explicit migration decisions for legacy durable replay traces."""

from enum import Enum

from pydantic import Field

from yewo.agent.memory.replay_store import MATCHER, SCHEMA, ReplayTrace
from yewo.discovery.artifacts import (
    DiscoveryCompatibilityKey,
    DiscoveryContract,
    SHA256_PATTERN,
)

REPLAY_MIGRATION_RULE_VERSION = "legacy-replay-v2-to-discovery-v1"


class ReplayMigrationDisposition(str, Enum):
    REVALIDATION_REQUIRED = "revalidation_required"
    REJECTED = "rejected"


class ReplayMigrationAssessment(DiscoveryContract):
    """What legacy mechanics may seed discovery without becoming trusted knowledge."""

    rule_version: str = REPLAY_MIGRATION_RULE_VERSION
    source_digest: str = Field(pattern=SHA256_PATTERN)
    disposition: ReplayMigrationDisposition
    reusable_transition_positions: tuple[int, ...] = ()
    reasons: tuple[str, ...] = Field(min_length=1)


def assess_replay_migration(
    trace: ReplayTrace,
    compatibility: DiscoveryCompatibilityKey,
    source_digest: str,
) -> ReplayMigrationAssessment:
    """Classify a trace without copying scenario identity, values, or verdict proof."""
    reasons: list[str] = []
    if trace.schema_version != SCHEMA or trace.matcher_version != MATCHER:
        reasons.append("The replay schema or matcher is not the current legacy contract.")
    if trace.learned_from is None:
        reasons.append("The replay trace does not identify the app artifact it observed.")
    elif trace.learned_from != compatibility.app_artifact_sha256:
        reasons.append("The replay trace belongs to a different app artifact.")
    if reasons:
        return ReplayMigrationAssessment(
            source_digest=source_digest,
            disposition=ReplayMigrationDisposition.REJECTED,
            reasons=tuple(reasons),
        )

    reusable_positions = []
    for transition in trace.transitions:
        action = transition.action
        if action.kind != "tap":
            continue
        if action.value_name is not None or action.postconditions:
            continue
        if action.target is None:
            continue
        if transition.source.phash is None or transition.destination.phash is None:
            continue
        reusable_positions.append(transition.position)

    if not reusable_positions:
        return ReplayMigrationAssessment(
            source_digest=source_digest,
            disposition=ReplayMigrationDisposition.REJECTED,
            reasons=("No transition is a value-free, reacquirable navigation tap.",),
        )
    return ReplayMigrationAssessment(
        source_digest=source_digest,
        disposition=ReplayMigrationDisposition.REVALIDATION_REQUIRED,
        reusable_transition_positions=tuple(reusable_positions),
        reasons=(
            "Legacy replay lacks discovery authorization and settlement receipts. "
            "Its safe navigation taps may seed the frontier but require live revalidation.",
        ),
    )
