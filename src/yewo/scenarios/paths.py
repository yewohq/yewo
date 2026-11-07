"""Canonical scenario artifact paths beneath one run root."""

from pathlib import Path

from yewo.artifacts import ARTIFACTS_DIRECTORY, RecordIdentifier


SCENARIO_ARTIFACTS_DIRECTORY = ARTIFACTS_DIRECTORY / "scenarios"
SCENARIO_GENERATION_PATH = SCENARIO_ARTIFACTS_DIRECTORY / "generation.json"


def scenario_artifact_directory(scenario_id: RecordIdentifier) -> Path:
    """Return the artifact directory for one generated scenario."""
    return SCENARIO_ARTIFACTS_DIRECTORY / scenario_id


def scenario_execution_path(scenario_id: RecordIdentifier) -> Path:
    """Return the execution record path for one generated scenario."""
    return scenario_artifact_directory(scenario_id) / "execution.json"


def scenario_audit_path(scenario_id: RecordIdentifier) -> Path:
    """Return the audit record path for one generated scenario."""
    return scenario_artifact_directory(scenario_id) / "audit.json"


def scenario_child_artifact_path(
    scenario_id: RecordIdentifier,
    child_relative_path: Path,
) -> Path:
    """Return one child-workspace artifact path from the run root."""
    return scenario_artifact_directory(scenario_id) / child_relative_path
