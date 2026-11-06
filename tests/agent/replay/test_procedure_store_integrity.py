"""Compiled-procedure snapshot integrity."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.discovery.records import DiscoveryObjective
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure.preparation import prepare_replay_procedure
from yewo.agent.replay.procedure.store import (
    CompiledProcedureSchemaError,
    CompiledProcedureStore,
    CompiledProcedureStoreError,
)

from agent.replay.support import evidence as _evidence, response as _response


def _rewrite_as_previous_schema(path: Path) -> None:
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload["bundle"]["schema_version"] = 1
    payload["bundle"]["procedure"]["schema_version"] = 1
    payload["bundle"]["procedure"].pop("setup_contract")
    content = json.dumps(
        payload["bundle"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    payload["bundle_sha256"] = sha256(content).hexdigest()
    path.write_text(json.dumps(payload))


def test_compiled_procedure_store_rejects_tampered_snapshot(tmp_path: Path) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path)
    store.save(compatibility, evidence, procedure)
    path = next((tmp_path / "procedures").rglob("procedure.json"))
    path.write_bytes(path.read_bytes().replace(b'"procedure_id":', b'"changed_id":', 1))

    with pytest.raises(CompiledProcedureStoreError):
        store.load(
            compatibility,
            evidence.task.objective_id,
            evidence.task.objective,
        )


def test_previous_procedure_schema_is_rebuilt_instead_of_aborting_preparation(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path)
    store.save(compatibility, evidence, procedure)
    path = next((tmp_path / "procedures").rglob("procedure.json"))
    _rewrite_as_previous_schema(path)

    with pytest.raises(CompiledProcedureSchemaError, match="Rebuild the procedure"):
        store.load(compatibility, evidence.task.objective_id, evidence.task.objective)

    preparation = prepare_replay_procedure(
        DiscoveryObjective(
            objective_id=evidence.task.objective_id,
            statement=evidence.task.objective,
            completion_conditions=evidence.task.completion_conditions,
        ),
        compatibility,
        None,
        store,
        lambda: pytest.fail("Stale exact procedures must not invoke objective matching."),
        "replay-preparer",
        maximum_model_requests=1,
    )

    assert preparation.selection is None


def test_stale_candidate_does_not_hide_valid_procedure(tmp_path: Path) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path)
    store.save(compatibility, evidence, procedure)
    stale_path = next((tmp_path / "procedures").rglob("procedure.json"))
    _rewrite_as_previous_schema(stale_path)
    valid_evidence = evidence.model_copy(
        update={
            "task": evidence.task.model_copy(
                update={"objective_id": "objective_2", "objective": "Save another item."}
            )
        }
    )
    valid_procedure = procedure.model_copy(update={"procedure_id": "procedure_2"})
    store.save(compatibility, valid_evidence, valid_procedure)

    candidates = store.compatible_candidates(compatibility)

    assert tuple(item.procedure.procedure_id for item in candidates) == ("procedure_2",)
