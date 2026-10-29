"""Prove one run began with no prior lock, knowledge, or saved path.

A warm run and a cold run write the same kind of report. Only a recorded,
enforced starting state tells them apart afterwards, so this module refuses
the run rather than describing an isolation it did not get.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from yewo.errors import UsageError
from yewo.testing.lock_file import path_beside


class ColdStartEvidence(BaseModel):
    """The isolation one cold run was granted, as its record keeps it."""

    knowledge_root: str
    output_root: str
    prior_artifact_count: int
    durable_replay_disabled: bool
    app_state_cleared: bool


def _existing_artifact_paths(root: Path) -> tuple[Path, ...]:
    """Every sealed artifact already under a knowledge root, by any layout."""
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob("*.json") if path.is_file()))


def require_clean_room(
    journey_file: Path | None,
    knowledge_root: Path,
    output_root: Path,
    *,
    replay: bool,
    durable_replay: bool,
    reuse_app: bool,
    locked: bool,
) -> ColdStartEvidence:
    """Refuse a cold run that could read a prior artifact, and record what it got.

    Args:
        journey_file: The journey this run was asked for. Its directory is
            where a reviewed lock would be.
        knowledge_root: Where this run may read and write app knowledge.
        output_root: Where this run writes its artifacts.
        replay: Whether in-run navigation reuse is on.
        durable_replay: Whether saved-path reuse is on.
        reuse_app: Whether the app keeps the state an earlier run left.
        locked: Whether the run was asked to execute a reviewed lock.

    Returns:
        The evidence a cold manifest carries.

    Raises:
        UsageError: If a lock, stored knowledge, or prior output exists, or if
            any form of reuse is still enabled.
    """
    if locked:
        raise UsageError(
            "A cold run generates its own suite. Drop --locked, or drop --cold to "
            "run the reviewed lock."
        )
    if replay or durable_replay:
        raise UsageError(
            "A cold run cannot reuse navigation. Remove --replay and pass "
            "--no-durable-replay."
        )
    if reuse_app:
        raise UsageError("A cold run cannot keep app state. Remove --reuse-app.")
    lock_path = path_beside(journey_file) if journey_file is not None else None
    if lock_path is not None and lock_path.exists():
        raise UsageError(
            f"A cold run cannot start beside a reviewed lock. Move or remove {lock_path}."
        )
    stored = _existing_artifact_paths(knowledge_root)
    if stored:
        raise UsageError(
            f"A cold run needs an empty knowledge root. {knowledge_root} already "
            f"holds {len(stored)} artifact(s), beginning with {stored[0].name}."
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise UsageError(
            f"A cold run needs an unused output root. {output_root} is not empty."
        )
    return ColdStartEvidence(
        knowledge_root=str(knowledge_root),
        output_root=str(output_root),
        prior_artifact_count=0,
        durable_replay_disabled=True,
        app_state_cleared=True,
    )
