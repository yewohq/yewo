"""Safe persistence and loading for model-call evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactRetention,
    ArtifactWriteError,
    new_record_identifier,
)
from yewo.devices.screen import png_size
from yewo.models.records import (
    ModelArtifactKind,
    ModelArtifactReference,
    ModelCallDisposition,
    ModelCallRecord,
    ModelRequest,
)
from yewo.observations import ScreenshotReference


class ModelArtifactError(RuntimeError):
    """Raised when model-call evidence cannot be handled safely."""


@dataclass(frozen=True)
class ModelArtifactPolicy:
    """Declare how one run persists model request and response content."""

    retention: ArtifactRetention
    retain_request_bodies: bool = True


@dataclass(frozen=True)
class ModelResponseEvidence:
    """Provider response content ready for one call ledger row."""

    body: dict[str, object]
    body_sha256: str
    content_sha256: str


class ModelArtifactStore:
    """Persist model requests and calls verbatim in append-only ledgers.

    A declared value reaches the model as its name. It becomes a literal only at the device
    boundary, so a recorded request holds no secret and needs no rewriting.
    """

    def __init__(
        self,
        root: Path,
        policy: ModelArtifactPolicy,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self.root = root
        self._policy = policy
        self._identifier_factory = identifier_factory
        self._writer = AppendOnlyJsonlWriter(root)

    def persist_request(
        self,
        request: ModelRequest,
        messages: list[dict[str, object]],
        response_format: dict[str, object] | None,
        provider_arguments: dict[str, object],
    ) -> ModelArtifactReference:
        """Persist the exact prepared request without embedded image bytes."""
        if request.retention is not self._policy.retention:
            raise ModelArtifactError("The model request exceeds the artifact policy.")
        request_content = {
            "messages": messages,
            "response_format": response_format,
        }
        request_options = {
            name: value for name, value in provider_arguments.items() if name != "messages"
        }
        body = {
            "request_id": request.request_id,
            "profile_id": request.profile_id,
            "role": request.role.value,
            "messages": messages,
            "request_options": request_options,
        }
        metadata = {
            **body,
            "messages": [self._message_metadata(message) for message in messages],
            "request_options": None,
            "request_options_sha256": _payload_digest(request_options),
        }
        return self._persist(
            ModelArtifactKind.REQUEST,
            metadata,
            body,
            content_sha256=_payload_digest(request_content),
            retain_body=self._policy.retain_request_bodies,
        )

    def prepare_response(
        self,
        *,
        request_id: str,
        attempt_number: int,
        disposition: ModelCallDisposition,
        text: str | None,
        reason: str | None,
        provider_payload: dict[str, object] | None = None,
    ) -> ModelResponseEvidence:
        """Prepare one response for inclusion in its call row."""
        body = {
            "request_id": request_id,
            "attempt_number": attempt_number,
            "disposition": disposition.value,
            "text": text,
            "reason": reason,
            "provider_payload": provider_payload,
        }
        return ModelResponseEvidence(
            body=body,
            body_sha256=_payload_digest(body),
            content_sha256=_payload_digest(
                {
                    "disposition": disposition.value,
                    "text": text,
                    "reason": reason,
                    "provider_payload": provider_payload,
                }
            ),
        )

    def persist_call(
        self,
        call: ModelCallRecord,
        response: ModelResponseEvidence,
    ) -> ModelArtifactReference:
        """Persist one provider attempt with its response content."""
        body = {
            **call.model_dump(mode="json", exclude={"call_artifact"}),
            "response": response.body,
            "response_body_sha256": response.body_sha256,
            "response_content_sha256": response.content_sha256,
        }
        return self._persist(
            ModelArtifactKind.CALL,
            body,
            body,
            content_sha256=_payload_digest(body),
            retain_body=True,
        )

    @staticmethod
    def _message_metadata(message: dict[str, object]) -> dict[str, object]:
        content = message.get("content")
        if isinstance(content, str):
            return {
                "role": message.get("role"),
                "content_sha256": _text_digest(content),
            }
        if not isinstance(content, list):
            raise ModelArtifactError("A prepared model message has invalid content.")
        return {
            "role": message.get("role"),
            "content": [ModelArtifactStore._content_metadata(part) for part in content],
        }

    @staticmethod
    def _content_metadata(content_body: object) -> object:
        if not isinstance(content_body, dict):
            raise ModelArtifactError("Model message content has an invalid structure.")
        if content_body.get("type") != "text":
            return content_body
        text = content_body.get("text")
        if not isinstance(text, str):
            raise ModelArtifactError("Model text content has an invalid structure.")
        return {"type": "text", "text_sha256": _text_digest(text)}

    def _persist(
        self,
        kind: ModelArtifactKind,
        metadata: dict[str, object],
        body: dict[str, object],
        *,
        content_sha256: str,
        retain_body: bool,
    ) -> ModelArtifactReference:
        artifact_id = self._identifier_factory()
        body_sha256 = _payload_digest(body)
        persisted_record = {
            "artifact_id": artifact_id,
            "kind": kind.value,
            "body_retained": retain_body,
            "retention": self._policy.retention.value,
            "body_sha256": body_sha256,
            "content_sha256": content_sha256,
            **(body if retain_body else metadata),
        }
        relative_path = ARTIFACTS_DIRECTORY / "models" / f"{kind.value}s.jsonl"
        try:
            content = self._writer.append(
                relative_path,
                persisted_record,
                identity_field="artifact_id",
            )
            reference = ModelArtifactReference(
                artifact_id=artifact_id,
                kind=kind,
                sha256=sha256(content).hexdigest(),
                body_sha256=body_sha256,
                content_sha256=content_sha256,
                path=relative_path,
                body_retained=retain_body,
                retention=self._policy.retention,
            )
        except (ArtifactWriteError, ValueError) as error:
            raise ModelArtifactError("Could not persist model-call evidence.") from error
        return reference


class ModelImageReader:
    """Load a screenshot after verifying its durable identity."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read(self, reference: ScreenshotReference) -> bytes:
        """Return verified PNG bytes from one run workspace."""
        path = self.root / reference.path
        try:
            path.resolve().relative_to(self.root.resolve())
            if path.is_symlink():
                raise ModelArtifactError("A model image cannot be a symbolic link.")
            content = path.read_bytes()
        except ModelArtifactError:
            raise
        except (OSError, ValueError) as error:
            raise ModelArtifactError("Could not read the model image artifact.") from error
        if sha256(content).hexdigest() != reference.sha256:
            raise ModelArtifactError("The model image digest does not match its reference.")
        if png_size(content) != (reference.width_px, reference.height_px):
            raise ModelArtifactError("The model image dimensions do not match its reference.")
        return content


def _text_digest(text: str | None) -> str | None:
    return sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def _payload_digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
