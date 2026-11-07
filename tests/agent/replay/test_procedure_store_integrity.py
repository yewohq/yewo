"""Compiled-procedure snapshot integrity."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic import ValidationError

from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.discovery.records import DiscoveryObjective
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure.preparation import prepare_replay_procedure
from yewo.agent.replay.procedure.store import (
    CompiledProcedureSchemaError,
    CompiledProcedureStore,
    CompiledProcedureStoreError,
    StoredProcedureBundle,
)

from agent.replay.support import evidence as _evidence, response as _response


def _evidence_with_screenshots(source_root: Path):
    evidence = _evidence()
    states = {}
    for index, (state_id, state) in enumerate(evidence.states.items(), start=1):
        reference = state.representative_screenshot
        stream = BytesIO()
        Image.new(
            "RGB",
            (reference.width_px, reference.height_px),
            color=(index * 20, 40, 80),
        ).save(stream, format="PNG")
        content = stream.getvalue()
        reference = reference.model_copy(update={"sha256": sha256(content).hexdigest()})
        path = source_root / reference.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        states[state_id] = state.model_copy(update={"representative_screenshot": reference})
    return evidence.model_copy(update={"states": states})


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


def _rewrite_with_previous_evidence_schema(path: Path) -> None:
    payload = json.loads(path.read_text())
    payload["bundle"]["evidence"]["schema_version"] = 1
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


def test_compiled_procedure_store_owns_and_stages_evidence_screenshots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-run"
    evidence = _evidence_with_screenshots(source_root)
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path / "durable")

    bundle = store.save(
        compatibility,
        evidence,
        procedure,
        source_run_root=source_root,
    )
    destination_root = tmp_path / "replay-run"
    store.stage_screenshots(bundle, destination_root)
    store.stage_screenshots(bundle, destination_root)

    references = tuple(state.representative_screenshot for state in evidence.states.values())
    assert all(
        (destination_root / reference.path).read_bytes()
        == (source_root / reference.path).read_bytes()
        for reference in references
    )


def test_compiled_procedure_store_repeated_save_reuses_identical_screenshots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-run"
    evidence = _evidence_with_screenshots(source_root)
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path / "durable")

    first = store.save(
        compatibility,
        evidence,
        procedure,
        source_run_root=source_root,
    )
    second = store.save(
        compatibility,
        evidence,
        procedure,
        source_run_root=source_root,
    )

    assert second == first


def test_compiled_procedure_store_rejects_missing_source_screenshot(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path / "durable")

    with pytest.raises(CompiledProcedureStoreError, match="regular file"):
        store.save(
            compatibility,
            evidence,
            procedure,
            source_run_root=tmp_path / "missing-run",
        )


def test_bundle_rejects_an_unknown_completion_condition_index() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    invalid_step = procedure.steps[0].model_copy(
        update={"captures_completion_conditions": (len(evidence.task.completion_conditions),)}
    )
    procedure = procedure.model_copy(update={"steps": (invalid_step, *procedure.steps[1:])})

    with pytest.raises(ValidationError, match="completion-condition references"):
        StoredProcedureBundle(
            compatibility=MemoryCompatibility(application=evidence.run.application),
            evidence=evidence,
            procedure=procedure,
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


def test_previous_evidence_schema_requests_a_rebuild(tmp_path: Path) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    compatibility = MemoryCompatibility(application=evidence.run.application)
    store = CompiledProcedureStore(tmp_path)
    store.save(compatibility, evidence, procedure)
    path = next((tmp_path / "procedures").rglob("procedure.json"))
    _rewrite_with_previous_evidence_schema(path)

    with pytest.raises(CompiledProcedureSchemaError, match="Rebuild the procedure"):
        store.load(compatibility, evidence.task.objective_id, evidence.task.objective)


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
