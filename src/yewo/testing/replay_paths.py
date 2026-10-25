"""Write policy for repository-owned replay paths."""

import logging
import os
import sys
from pathlib import Path

from yewo.agent.memory import replay_files, replay_store
from yewo.errors import UsageError

logger = logging.getLogger(__name__)

CI_ENVIRONMENT_VARIABLES = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "TF_BUILD",
)


def is_ci() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() not in ("", "0", "false")
        for name in CI_ENVIRONMENT_VARIABLES
    )


def validate_path_key(root: Path) -> None:
    """Make repository replay behavior explicit before a run spends money."""
    if os.environ.get(replay_store.PATH_KEY_ENV):
        return
    path_root = root / replay_files.STORE_DIRECTORY
    has_saved_paths = path_root.exists() and any(path_root.rglob("*.json"))
    if is_ci() and has_saved_paths:
        raise UsageError(
            f"{path_root} contains saved replay paths, but "
            f"{replay_store.PATH_KEY_ENV} is not set. Configure the same secret "
            "in CI that created these paths, or run with --no-durable-replay."
        )
    if not is_ci():
        logger.warning(
            "%s is not set. Saved replay paths will match only on this machine.",
            replay_store.PATH_KEY_ENV,
        )


def approve_update(path: Path, update_paths: bool) -> bool:
    if is_ci():
        logger.info("Replay path was not updated: CI runs are read-only.")
        return False
    if update_paths:
        return True
    if not sys.stdin.isatty():
        logger.info(
            "Replay path changed but was not updated. Re-run with --update-paths "
            "to accept it."
        )
        return False
    return input(f"Update replay path {path}? [y/N] ").strip().lower() in ("y", "yes")
