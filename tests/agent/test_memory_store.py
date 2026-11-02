"""YewoAgent persistent app-memory tests."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from yewo.agent import (
    AppMemory,
    AppMemoryStore,
    AppMemoryStoreError,
    MemoryCompatibility,
    UnsuccessfulAttempt,
    UnsuccessfulAttemptDisposition,
)
from yewo.agent.actuation import ActionKind, ActionOutcomeSignal
from yewo.artifacts import ArtifactRetention
from yewo.grounding import PixelPoint
from yewo.agent.discovery import (
    AppMemoryBuilder,
    DiscoveryObjective,
    DiscoveryRequest,
)
from yewo.run_narration import RunNarrator, narration_scope

from agent.discovery.support import agent_result, observation


def _memory(source_root: Path) -> tuple[MemoryCompatibility, AppMemory]:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    png = _png()
    screenshot_sha256 = sha256(png).hexdigest()
    before = _write_screenshot(source_root, before, png, screenshot_sha256)
    after = _write_screenshot(source_root, after, png, screenshot_sha256)
    compatibility = MemoryCompatibility(application=before.application)
    promotion = AppMemoryBuilder().promote(
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="account_form",
                    statement="The account form is reachable.",
                ),
            ),
        ),
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        compatibility,
        verified_objective_ids=frozenset({"account_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return compatibility, promotion.memory


def _write_screenshot(source_root, source, png, screenshot_sha256):
    screenshot = source.redacted_screenshot.model_copy(update={"sha256": screenshot_sha256})
    path = source_root / screenshot.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return source.model_copy(
        update={
            "redacted_screenshot": screenshot,
            "screen_fingerprint": source.screen_fingerprint.model_copy(
                update={"source_artifact_id": screenshot.artifact_id}
            ),
        }
    )


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (200, 400), color=(10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _stored_application_root(store: AppMemoryStore) -> Path:
    return next((store.root / "applications").iterdir())


def test_store_promotes_redacted_screenshots_and_loads_verified_memory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")

    with narration_scope(RunNarrator(tmp_path)):
        persisted = store.save(memory, source_root)
        loaded = store.load(compatibility)

    assert loaded == persisted
    assert {
        variant.screenshot.retention for screen in persisted.screens for variant in screen.variants
    } == {ArtifactRetention.PERSISTENT}
    assert all(
        variant.screenshot.path.parts[0] == "screenshots"
        for screen in persisted.screens
        for variant in screen.variants
    )
    assert (_stored_application_root(store) / "memory.json").is_file()
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Saved application memory with 1 capability record(s)." in narration
    assert "Loaded application memory with 1 capability record(s)." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert f"save=completed memory={persisted.memory_id}" in trace
    assert f"load=completed memory={persisted.memory_id}" in trace


def test_store_rejects_a_changed_snapshot_digest(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, source_root)
    path = _stored_application_root(store) / "memory.json"
    envelope = json.loads(path.read_text())
    envelope["memory_sha256"] = "0" * 64
    path.write_text(json.dumps(envelope))

    with pytest.raises(AppMemoryStoreError, match="digest"):
        store.load(compatibility)


def test_store_loads_a_digest_created_before_optional_memory_fields(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, source_root)
    path = _stored_application_root(store) / "memory.json"
    envelope = json.loads(path.read_text())
    legacy_memory = envelope["memory"]
    legacy_memory.pop("checkpoints")
    content = json.dumps(
        legacy_memory,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    envelope["memory_sha256"] = sha256(content).hexdigest()
    path.write_text(json.dumps(envelope))

    loaded = store.load(compatibility)

    assert loaded is not None
    assert loaded.checkpoints == ()


def test_store_sorts_outcome_signals_before_hashing_memory(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    transition = memory.transitions[0].model_copy(
        update={
            "outcome_signals": frozenset(
                {
                    ActionOutcomeSignal.SCREEN_CHANGED,
                    ActionOutcomeSignal.CONTEXT_CHANGED,
                }
            )
        }
    )
    memory = memory.model_copy(update={"transitions": (transition,)})
    store = AppMemoryStore(tmp_path / "shared")

    store.save(memory, source_root)
    envelope = json.loads((_stored_application_root(store) / "memory.json").read_text())

    assert envelope["memory"]["transitions"][0]["outcome_signals"] == [
        "context_changed",
        "screen_changed",
    ]
    assert store.load(compatibility) is not None


def test_store_invalidates_legacy_target_rejection_evidence_on_load(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    transition = memory.transitions[0]
    source_screen = next(
        screen for screen in memory.screens if screen.screen_id == transition.source_screen_id
    )
    source_variant = source_screen.variants[0]
    attempt = UnsuccessfulAttempt(
        attempt_id="attempt_1",
        screen_id=source_screen.screen_id,
        source_observation_id=source_variant.observation_id,
        action=ActionKind.TAP,
        target_description="the add button",
        disposition=UnsuccessfulAttemptDisposition.TARGET_REJECTED,
        reason="The grounded point selected a label.",
        grounded_point=PixelPoint(x=100, y=200),
        screenshot_width_px=source_variant.width_px,
        screenshot_height_px=source_variant.height_px,
        grounding_profile_id=transition.grounding_profile_ids[0],
        grounder_model_profile_id=transition.grounder_model_profile_ids[0],
        requested_grounder_model=transition.requested_grounder_models[0],
        source_step_id="step_1",
        source_run_id="run_1",
    )
    memory = AppMemory.model_validate({**memory.model_dump(), "unsuccessful_attempts": (attempt,)})
    store = AppMemoryStore(tmp_path / "shared")

    persisted = store.save(memory, source_root)
    loaded = store.load(compatibility)

    assert persisted.unsuccessful_attempts == (attempt,)
    assert loaded is not None
    assert loaded.unsuccessful_attempts == ()


def test_store_rejects_a_changed_persistent_screenshot(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    persisted = store.save(memory, source_root)
    screenshot = persisted.screens[0].variants[0].screenshot
    (_stored_application_root(store) / screenshot.path).write_bytes(b"changed")

    with pytest.raises(AppMemoryStoreError, match="digest"):
        store.load(compatibility)


def test_store_rejects_a_stale_memory_replacement(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    persisted = store.save(memory, source_root)
    stale = persisted.model_copy(update={"source_run_ids": ("run_2",)})

    with pytest.raises(AppMemoryStoreError, match="stale source runs"):
        store.save(stale, source_root)


def test_store_removes_screenshots_no_longer_referenced_by_the_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "run"
    unused_compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    persisted = store.save(memory, source_root)
    application_root = _stored_application_root(store)
    orphan = application_root / "screenshots/orphan.png"
    orphan.write_bytes(b"obsolete")

    store.save(persisted, source_root)

    assert not orphan.exists()
