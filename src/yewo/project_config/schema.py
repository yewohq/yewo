"""Publish the JSON Schema for a yewo.yaml from the models that validate one."""

from __future__ import annotations

import json

from yewo.project_config.records import YewoFileSpec


SCHEMA_DIALECT = "http://json-schema.org/draft-07/schema#"
SCHEMA_ID = "https://raw.githubusercontent.com/yewohq/yewo/master/yewo-schema.json"


def build_project_schema() -> str:
    """Render the published schema from the declaration model an editor must match.

    An editor reads the schema and a run reads the model. Rendering one from the other keeps
    an editor from offering a key a run refuses, or refusing a key a run accepts.
    """
    schema = {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        **YewoFileSpec.model_json_schema(),
    }
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


__all__ = ["SCHEMA_DIALECT", "SCHEMA_ID", "build_project_schema"]
