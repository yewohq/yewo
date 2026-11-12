"""The published schema an editor reads must match the models a run enforces."""

from __future__ import annotations

import json
from pathlib import Path

from yewo.project_config.records import YewoFileSpec
from yewo.project_config.schema import build_project_schema


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "yewo-schema.json"


def test_the_committed_schema_matches_the_declaration_model() -> None:
    """The file drifted into describing a model release that no longer exists.

    It still offered `perception_model` and `grounding`, which a run now refuses, and omitted
    `models`, `grounder` and `memory_root`, which a run now accepts. Both directions are a
    broken editor, because the schema forbids every property it does not list.
    """
    assert SCHEMA_PATH.read_text(encoding="utf-8") == build_project_schema(), (
        "yewo-schema.json is out of date. Run: uv run python tools/emit_schema.py"
    )


def test_the_schema_offers_exactly_the_declared_settings() -> None:
    """Derived from the model, so a hand-edited schema fails this without the generator."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(YewoFileSpec.model_fields)


def test_the_schema_carries_the_per_role_model_selection() -> None:
    """The configuration an editor could neither suggest nor accept."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    roles = schema["$defs"]["ModelOverridesSpec"]["properties"]

    assert {"proposal_verifier", "outcome_verifier", "screen_analyzer"} <= set(roles)
    assert "reasoning" in schema["$defs"]["ModelSelectionSpec"]["properties"]
    assert "protocol" in schema["$defs"]["GrounderSelectionSpec"]["properties"]
