"""Where a judged path is kept between runs, and what is refused on reading.

One file per scenario per target. A trace is an optimisation, so every
failure here is a refusal rather than an error. A run that cannot read its
cache plans the scenario, which is what it did before this phase.

Nothing is written until a scenario has been judged and passed. A path
that ended in a failure is a path to a failure, and replaying it would
spend a run reaching a result already known.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from yewo.agent.memory.replay_store import MATCHER, SCHEMA, ReplayTrace, TraceIdentity

logger = logging.getLogger(__name__)

STORE_DIRECTORY = ".yewo/paths"
"""Beside the journey file, not in the run's artifacts.

A run's artifacts belong to that run. This outlives it on purpose, so it
    sits where the next run will look rather than under a session id nobody
    will guess. It is a project-owned path and may be reviewed in version
    control.
"""


def _as_one_name(value: str) -> str:
    """One path segment, from a string that reached this from a file.

    A uuid and a device name both arrive from `yewo.lock`, which a person
    edits. Either could name a parent directory, and a store that followed
    it would write outside itself.
    """
    safe_segment = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe_segment or "unknown"


def trace_path(root: Path, identity: TraceIdentity) -> Path:
    """The one file this scenario and this target own.

    The profile is in the name rather than in the file, so two targets of
    one scenario never open each other's path. Opening the wrong one would
    be found by the matcher, but only after it had been trusted.
    """
    return (
        root
        / STORE_DIRECTORY
        / f"v{SCHEMA}"
        / _as_one_name(identity.scenario_uuid)
        / f"{_as_one_name(identity.target_profile)}.json"
    )


def read_trace(path: Path, identity: TraceIdentity) -> Optional[ReplayTrace]:
    """The trace this run may use, or None with the reason logged.

    Every refusal is quiet in effect and loud in the log. The outcomes are
    one outcome: plan it. A corrupt file, a schema from another version, a
    stricter matcher and another scenario's trace all reach it.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring an unreadable replay trace: %s", type(exc).__name__)
        return None

    if not isinstance(payload, dict):
        logger.warning("Ignoring a replay trace that is not a record.")
        return None
    if payload.get("schema") != SCHEMA:
        logger.info("Ignoring a replay trace written for schema %s.", payload.get("schema"))
        return None
    if payload.get("matcher") != MATCHER:
        logger.info("Ignoring a replay trace matched by version %s.", payload.get("matcher"))
        return None

    try:
        trace = ReplayTrace.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Ignoring a malformed replay trace: %s", exc.error_count())
        return None

    if trace.identity != identity:
        logger.info("Ignoring a replay trace recorded for a different target.")
        return None
    return trace


def write_trace(path: Path, trace: ReplayTrace) -> bool:
    """Replaces the trace for this scenario, or says it could not.

    Written whole and moved into place. A reader of this file is a later
    run, and a half-written file is one a later run would have to refuse.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".part"
        ) as part_file_handle:
            json.dump(
                trace.model_dump(by_alias=True), part_file_handle, indent=2, sort_keys=True
            )
            part_file_path = Path(part_file_handle.name)
        os.replace(part_file_path, path)
        return True
    except OSError as exc:  # noqa: BLE001 - a cache must not fail a run
        logger.warning("Could not write the replay trace: %s", exc)
        return False
