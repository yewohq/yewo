"""Tests for atomic screenshot persistence."""

from hashlib import sha256
from pathlib import Path

import pytest

from symlink_support import requires_symlinks

from yewo.artifacts import ArtifactRetention
from yewo.observations import (
    Observation,
    ObservationCaptureError,
    ObservationStore,
    ScreenshotStore,
)
from observations.factories import build_observation
from support import png


def _persistable_observation(tmp_path: Path) -> Observation:
    identifiers = iter(("raw_1", "redacted_1"))
    store = ScreenshotStore(
        tmp_path,
        identifier_factory=lambda: next(identifiers),
    )
    store.persist(
        png(),
        retention=ArtifactRetention.RUN,
    )
    screenshot = store.persist(
        png(),
        retention=ArtifactRetention.RUN,
    )
    return build_observation(
        screenshot=screenshot,
    )


def test_store_persists_verified_png_under_relative_path(tmp_path: Path) -> None:
    content = png()
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")

    reference = store.persist(
        content,
        retention=ArtifactRetention.RUN,
    )

    assert reference.path == Path("screenshots/shot_1.png")
    assert reference.width_px == 3
    assert reference.height_px == 2
    assert reference.sha256 == sha256(content).hexdigest()
    assert (tmp_path / reference.path).read_bytes() == content
    assert tuple((tmp_path / "screenshots").iterdir()) == (tmp_path / "screenshots/shot_1.png",)


def test_store_rejects_invalid_png_without_writing(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")

    with pytest.raises(ObservationCaptureError, match="invalid PNG"):
        store.persist(
            b"not a PNG",
            retention=ArtifactRetention.RUN,
        )

    assert not (tmp_path / "screenshots").exists()


def test_store_does_not_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")
    arguments = {
        "retention": ArtifactRetention.RUN,
    }
    store.persist(png(), **arguments)

    with pytest.raises(ObservationCaptureError, match="already exists"):
        store.persist(png(), **arguments)

    assert (tmp_path / "screenshots/shot_1.png").read_bytes() == png()


def test_store_rejects_an_unsafe_generated_identifier(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "../escape")

    with pytest.raises(ObservationCaptureError, match="identifier or metadata"):
        store.persist(
            png(),
            retention=ArtifactRetention.RUN,
        )

    assert not (tmp_path / "screenshots").exists()


@requires_symlinks
def test_store_rejects_screenshot_directory_outside_artifact_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "screenshots").symlink_to(outside, target_is_directory=True)
    store = ScreenshotStore(artifact_root, identifier_factory=lambda: "shot_1")

    with pytest.raises(ObservationCaptureError, match="outside the artifact root"):
        store.persist(
            png(),
            retention=ArtifactRetention.RUN,
        )

    assert tuple(outside.iterdir()) == ()


def test_observation_store_persists_round_trippable_json(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)

    store = ObservationStore(tmp_path)
    relative_path = store.persist(observation)

    assert relative_path == Path("observations.jsonl")
    restored = observation.model_validate_json(
        (tmp_path / relative_path).read_text().splitlines()[0]
    )
    assert restored == observation
    assert store.load_all() == (observation,)


def test_observation_store_loads_only_requested_evidence(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)
    second = observation.model_copy(update={"observation_id": "observation_2"})
    store = ObservationStore(tmp_path)
    store.persist(observation)
    store.persist(second)

    assert store.load_selected({second.observation_id}) == (second,)

    with pytest.raises(ObservationCaptureError, match="missing requested"):
        store.load_selected({"missing_observation"})


def test_observation_store_rejects_an_invalid_ledger(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json\n")

    with pytest.raises(ObservationCaptureError, match="ledger is invalid"):
        ObservationStore(tmp_path).load_all()


def test_observation_store_rejects_an_incomplete_final_line(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)
    store = ObservationStore(tmp_path)
    relative_path = store.persist(observation)
    path = tmp_path / relative_path
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    with pytest.raises(ObservationCaptureError, match="incomplete final line"):
        store.load_all()


def test_observation_store_does_not_overwrite_existing_record(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path)
    observation = _persistable_observation(tmp_path)
    relative_path = store.persist(observation)
    original_content = (tmp_path / relative_path).read_bytes()

    with pytest.raises(ObservationCaptureError, match="already exists"):
        store.persist(observation)

    assert (tmp_path / relative_path).read_bytes() == original_content


def test_observation_store_rejects_missing_screenshot(tmp_path: Path) -> None:
    observation = build_observation()

    with pytest.raises(ObservationCaptureError, match="verify observation screenshot"):
        ObservationStore(tmp_path).persist(observation)

    assert not (tmp_path / "observations.jsonl").exists()


def test_observation_store_rejects_screenshot_digest_mismatch(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)
    assert observation.screenshot is not None
    (tmp_path / observation.screenshot.path).write_bytes(png() + b"changed")

    with pytest.raises(ObservationCaptureError, match="digest mismatch"):
        ObservationStore(tmp_path).persist(observation)

    assert not (tmp_path / "observations.jsonl").exists()
