"""Tests for safe model-call artifact handling."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from symlink_support import requires_symlinks

from yewo.artifacts import ArtifactRetention
from yewo.models import (
    ImageModelContent,
    ModelArtifactError,
    ModelArtifactKind,
    ModelArtifactPolicy,
    ModelArtifactStore,
    ModelCallDisposition,
    ModelCallRecord,
    ModelImageReader,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
)
from yewo.observations import ScreenshotStore
from support import png


def _screenshot(tmp_path: Path):
    return ScreenshotStore(tmp_path, identifier_factory=lambda: "shot-1").persist(
        png(),
        retention=ArtifactRetention.RUN,
    )


def _request(tmp_path: Path) -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        profile_id="grounder-primary",
        role=ModelRole.GROUNDER,
        messages=(
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(
                    TextModelContent(text="Submit hunter2 value"),
                    ImageModelContent(screenshot=_screenshot(tmp_path)),
                ),
            ),
        ),
        retention=ArtifactRetention.RUN,
    )


def _store(
    tmp_path: Path,
    *,
    retain_request_bodies: bool = True,
    identifiers: tuple[str, ...] = ("model-artifact-1",),
) -> ModelArtifactStore:
    identifier_iterator = iter(identifiers)
    return ModelArtifactStore(
        tmp_path,
        ModelArtifactPolicy(
            retention=ArtifactRetention.RUN,
            retain_request_bodies=retain_request_bodies,
        ),
        identifier_factory=lambda: next(identifier_iterator),
    )


def _prepared_messages(request: ModelRequest) -> list[dict[str, object]]:
    image = request.messages[0].content[1]
    assert isinstance(image, ImageModelContent)
    screenshot = image.screenshot
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Submit [yewo-redacted] value"},
                {
                    "type": "image_ref",
                    "artifact_id": screenshot.artifact_id,
                    "sha256": screenshot.sha256,
                    "width_px": screenshot.width_px,
                    "height_px": screenshot.height_px,
                },
            ],
        }
    ]


def _persist_request(
    store: ModelArtifactStore,
    request: ModelRequest,
):
    return store.persist_request(request, _prepared_messages(request), None, {})


def _persist_call(store: ModelArtifactStore, request: ModelRequest, prompt_artifact, **response):
    evidence = store.prepare_response(
        request_id=request.request_id,
        attempt_number=1,
        disposition=ModelCallDisposition.COMPLETED,
        reason=None,
        **response,
    )
    call = ModelCallRecord(
        request_id=request.request_id,
        profile_id=request.profile_id,
        role=request.role,
        requested_provider="openrouter",
        requested_model="example/model",
        attempt_number=1,
        started_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
        disposition=ModelCallDisposition.COMPLETED,
        prompt_artifact=prompt_artifact,
    )
    return store.persist_call(call, evidence)


def _record(path: Path, artifact_id: str) -> dict[str, object]:
    return next(
        payload
        for payload in (json.loads(line) for line in path.read_text().splitlines())
        if payload["artifact_id"] == artifact_id
    )


def test_request_body_retention_can_be_disabled(tmp_path: Path) -> None:
    request = _request(tmp_path)
    reference = _persist_request(
        _store(tmp_path, retain_request_bodies=False),
        request,
    )
    content = (tmp_path / reference.path).read_bytes()
    payload = json.loads(content)

    assert reference.kind is ModelArtifactKind.REQUEST
    assert reference.sha256 == sha256(content).hexdigest()
    assert payload["body_retained"] is False
    assert payload["messages"][0]["content"][0] == {
        "type": "text",
        "text_sha256": sha256(b"Submit [yewo-redacted] value").hexdigest(),
    }
    assert payload["messages"][0]["content"][1]["artifact_id"] == "shot-1"
    assert "hunter2" not in content.decode()
    assert "iVBOR" not in content.decode()


def test_request_and_call_bodies_are_retained_verbatim_by_default(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        identifiers=("request-artifact", "call-artifact"),
    )
    request = _request(tmp_path)
    request_reference = _persist_request(store, request)
    call_reference = _persist_call(
        store,
        request,
        request_reference,
        text="Tapped the Save control.",
        provider_payload={"choices": [{"message": {"content": "Tapped the Save control."}}]},
    )

    request_text = (tmp_path / request_reference.path).read_text()
    call_text = (tmp_path / call_reference.path).read_text()

    assert "hunter2" not in request_text
    assert "Submit [yewo-redacted] value" in request_text
    assert "Tapped the Save control." in call_text
    assert request_reference.body_retained is True
    assert call_reference.body_retained is True
    assert call_reference.kind is ModelArtifactKind.CALL


def test_call_response_is_retained_when_request_body_is_disabled(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        retain_request_bodies=False,
        identifiers=("request-artifact", "call-artifact"),
    )
    request = _request(tmp_path)
    request_reference = _persist_request(store, request)
    call_reference = _persist_call(
        store,
        request,
        request_reference,
        text="Tapped the Save control.",
        provider_payload={"content": "Tapped the Save control."},
    )

    call_payload = _record(tmp_path / call_reference.path, call_reference.artifact_id)

    assert request_reference.body_retained is False
    assert call_reference.body_retained is True
    response_payload = call_payload["response"]
    assert isinstance(response_payload, dict)
    assert response_payload["text"] == "Tapped the Save control."
    assert response_payload["provider_payload"] == {"content": "Tapped the Save control."}


def test_model_image_reader_verifies_digest_and_dimensions(tmp_path: Path) -> None:
    reference = _screenshot(tmp_path)

    assert ModelImageReader(tmp_path).read(reference) == png()

    (tmp_path / reference.path).write_bytes(png() + b"changed")
    with pytest.raises(ModelArtifactError, match="digest"):
        ModelImageReader(tmp_path).read(reference)


@requires_symlinks
def test_model_image_reader_rejects_symbolic_link(tmp_path: Path) -> None:
    reference = _screenshot(tmp_path)
    target = tmp_path / reference.path
    outside = tmp_path / "outside.png"
    outside.write_bytes(png())
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(ModelArtifactError, match="symbolic link"):
        ModelImageReader(tmp_path).read(reference)


def test_model_artifact_store_does_not_overwrite_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = _store(
        tmp_path,
        retain_request_bodies=False,
        identifiers=("duplicate", "duplicate"),
    )
    _persist_request(store, request)

    with pytest.raises(ModelArtifactError, match="persist"):
        _persist_request(store, request)
