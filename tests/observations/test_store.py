"""Tests for atomic screenshot persistence."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.observations import (
    Observation,
    ObservationCaptureError,
    ObservationStore,
    RedactionProvenance,
    ScreenshotRedactionState,
    ScreenshotStore,
)
from observations.factories import build_observation


REDACTION = RedactionProvenance(
    policy_id="region-mask",
    region_count=0,
    region_set_sha256="c" * 64,
    input_sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
)


def _png() -> bytes:
    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _persistable_observation(tmp_path: Path) -> Observation:
    identifiers = iter(("raw_1", "redacted_1"))
    store = ScreenshotStore(
        tmp_path,
        identifier_factory=lambda: next(identifiers),
    )
    raw_screenshot = store.persist_raw(
        _png(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    redacted_screenshot = store.persist_redacted(
        _png(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
        redaction=REDACTION,
    )
    return build_observation(
        raw_screenshot=raw_screenshot,
        redacted_screenshot=redacted_screenshot,
    )


def test_store_persists_verified_png_under_relative_path(tmp_path: Path) -> None:
    png = _png()
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")

    reference = store.persist_raw(
        png,
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )

    assert reference.path == Path("assets/screenshots/shot_1.png")
    assert reference.width_px == 3
    assert reference.height_px == 2
    assert reference.sha256 == sha256(png).hexdigest()
    assert (tmp_path / reference.path).read_bytes() == png
    assert tuple((tmp_path / "assets" / "screenshots").iterdir()) == (
        tmp_path / "assets" / "screenshots/shot_1.png",
    )


def test_store_rejects_invalid_png_without_writing(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")

    with pytest.raises(ObservationCaptureError, match="invalid PNG"):
        store.persist_raw(
            b"not a PNG",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )

    assert not (tmp_path / "assets").exists()


def test_store_marks_processed_screenshot_as_redacted(tmp_path: Path) -> None:
    reference = ScreenshotStore(
        tmp_path,
        identifier_factory=lambda: "redacted_1",
    ).persist_redacted(
        _png(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
        redaction=REDACTION,
    )

    assert reference.redaction_state is ScreenshotRedactionState.REDACTED
    assert reference.redaction == REDACTION


def test_store_does_not_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot_1")
    arguments = {
        "sensitivity": ArtifactSensitivity.DEVICE_CONTENT,
        "retention": ArtifactRetention.RUN,
    }
    store.persist_raw(_png(), **arguments)

    with pytest.raises(ObservationCaptureError, match="already exists"):
        store.persist_raw(_png(), **arguments)

    assert (tmp_path / "assets/screenshots/shot_1.png").read_bytes() == _png()


def test_store_rejects_an_unsafe_generated_identifier(tmp_path: Path) -> None:
    store = ScreenshotStore(tmp_path, identifier_factory=lambda: "../escape")

    with pytest.raises(ObservationCaptureError, match="identifier or metadata"):
        store.persist_raw(
            _png(),
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )

    assert not (tmp_path / "assets").exists()


def test_store_rejects_screenshot_directory_outside_artifact_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "assets").symlink_to(outside, target_is_directory=True)
    store = ScreenshotStore(artifact_root, identifier_factory=lambda: "shot_1")

    with pytest.raises(ObservationCaptureError, match="outside the artifact root"):
        store.persist_raw(
            _png(),
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )

    assert tuple(outside.iterdir()) == ()


def test_observation_store_persists_round_trippable_json(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)

    relative_path = ObservationStore(tmp_path).persist(observation)

    assert relative_path == Path("assets/observations.jsonl")
    restored = observation.model_validate_json((tmp_path / relative_path).read_text().splitlines()[0])
    assert restored == observation


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

    assert not (tmp_path / "assets/observations.jsonl").exists()


def test_observation_store_rejects_screenshot_digest_mismatch(tmp_path: Path) -> None:
    observation = _persistable_observation(tmp_path)
    assert observation.raw_screenshot is not None
    (tmp_path / observation.raw_screenshot.path).write_bytes(_png() + b"changed")

    with pytest.raises(ObservationCaptureError, match="digest mismatch"):
        ObservationStore(tmp_path).persist(observation)

    assert not (tmp_path / "assets/observations.jsonl").exists()
