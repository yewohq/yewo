"""Provider-neutral records for model requests and call evidence."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yewo.artifacts import (
    RelativeArtifactPath,
    ARTIFACTS_DIRECTORY,
    ArtifactRetention,
    ArtifactSensitivity,
    RecordIdentifier,
    Sha256Digest,
)
from yewo.observations import ScreenshotRedactionState, ScreenshotReference


class ModelRole(StrEnum):
    """A product responsibility assigned to one model call."""

    TASK_PLANNER = "task_planner"
    GROUNDER = "grounder"
    AUDITOR = "auditor"
    RECOVERY_ADVISOR = "recovery_advisor"
    ACTION_VERIFIER = "action_verifier"
    SCREEN_ANALYZER = "screen_analyzer"
    REPLAY_PREPARER = "replay_preparer"
    TEST_DESIGNER = "test_designer"


class ModelMessageRole(StrEnum):
    """The conversational role of one model message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ModelCallDisposition(StrEnum):
    """The normalized outcome of one model call."""

    COMPLETED = "completed"
    EMPTY = "empty"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    REASONING_EXHAUSTED = "reasoning_exhausted"
    MALFORMED = "malformed"
    PROVIDER_ERROR = "provider_error"


class ModelReasoningEffort(StrEnum):
    """Provider-neutral reasoning effort for one model request."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DEFAULT = "default"


class ModelArtifactKind(StrEnum):
    """The model boundary that owns one persisted artifact."""

    REQUEST = "request"
    CALL = "call"


class ModelArtifactReference(BaseModel):
    """Identity, policy, and location of one model-call artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: RecordIdentifier
    kind: ModelArtifactKind
    sha256: Sha256Digest
    body_sha256: Sha256Digest
    content_sha256: Sha256Digest
    path: RelativeArtifactPath
    body_retained: bool
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        if (
            value.is_absolute()
            or ".." in value.parts
            or value.name in {"", "."}
            or value.suffix != ".jsonl"
            or value.parts[: len(ARTIFACTS_DIRECTORY.parts) + 1]
            != (*ARTIFACTS_DIRECTORY.parts, "models")
        ):
            raise ValueError("A model artifact must use a safe JSONL path under models.")
        return value


class TextModelContent(BaseModel):
    """One text part sent to a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ImageModelContent(BaseModel):
    """One redacted screenshot part sent to a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["image"] = "image"
    screenshot: ScreenshotReference

    @model_validator(mode="after")
    def _validate_redacted_screenshot(self) -> Self:
        if self.screenshot.redaction_state is not ScreenshotRedactionState.REDACTED:
            raise ValueError("A model image must reference a redacted screenshot.")
        return self


ModelContent = Annotated[
    TextModelContent | ImageModelContent,
    Field(discriminator="kind"),
]


class ModelMessage(BaseModel):
    """One ordered message containing text, images, or both."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ModelMessageRole
    content: tuple[ModelContent, ...] = Field(min_length=1)
    cache_prefix_end: bool = False


class ModelResponseContract(BaseModel):
    """The stable local schema requested from a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: RecordIdentifier
    json_schema: dict[str, object]
    schema_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_schema_digest(self) -> Self:
        if self.schema_sha256 != _canonical_digest(self.json_schema):
            raise ValueError("The response-contract digest does not match its schema.")
        return self


def build_model_response_contract(
    name: str,
    json_schema: dict[str, object],
) -> ModelResponseContract:
    """Build a response contract with a canonical schema digest."""
    return ModelResponseContract(
        name=name,
        json_schema=json_schema,
        schema_sha256=_canonical_digest(json_schema),
    )


class ModelRequest(BaseModel):
    """One provider-neutral request bound to a selected profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: RecordIdentifier
    profile_id: RecordIdentifier
    role: ModelRole
    max_completion_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: ModelReasoningEffort | None = None
    retry_reasoning_exhaustion: bool = True
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    response_contract: ModelResponseContract | None = None
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention


class ModelCallUsage(BaseModel):
    """Normalized usage reported for one actual provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    prompt_image_tokens: int | None = Field(default=None, ge=0)
    prompt_text_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_details: dict[str, object] = Field(default_factory=dict)


class ModelCallRecord(BaseModel):
    """Immutable evidence for one actual call through the model transport."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: RecordIdentifier
    profile_id: RecordIdentifier
    role: ModelRole
    requested_provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    requested_revision: str | None = Field(default=None, min_length=1)
    served_model: str | None = Field(default=None, min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    provider_request_id: str | None = Field(default=None, min_length=1)
    attempt_number: int = Field(ge=1)
    validation_attempt_number: int = Field(default=1, ge=1)
    provider_attempt_number: int = Field(default=1, ge=1)
    started_at: datetime
    latency_ms: float | None = Field(default=None, ge=0)
    finish_reason: str | None = Field(default=None, min_length=1)
    disposition: ModelCallDisposition
    usage: ModelCallUsage = ModelCallUsage()
    prompt_artifact: ModelArtifactReference | None = None
    call_artifact: ModelArtifactReference | None = None

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("A model-call start time must include a timezone.")
        return value

    @model_validator(mode="after")
    def _validate_artifact_kinds(self) -> Self:
        if (
            self.prompt_artifact is not None
            and self.prompt_artifact.kind is not ModelArtifactKind.REQUEST
        ):
            raise ValueError("A prompt artifact must contain a model request.")
        if self.call_artifact is not None and self.call_artifact.kind is not ModelArtifactKind.CALL:
            raise ValueError("A call artifact must contain a model call.")
        return self


class ModelResponse(BaseModel):
    """The normalized final response for one logical model request."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: RecordIdentifier
    disposition: ModelCallDisposition
    text: str | None = Field(default=None, min_length=1)
    structured_output: dict[str, object] | None = None
    reason: str | None = Field(default=None, min_length=1)
    calls: tuple[ModelCallRecord, ...] = Field(min_length=1)
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention

    @model_validator(mode="after")
    def _validate_response(self) -> Self:
        if any(call.request_id != self.request_id for call in self.calls):
            raise ValueError("Every model call must match the response request.")
        if self.calls[-1].disposition is not self.disposition:
            raise ValueError("The final model call must match the response disposition.")
        if self.disposition is ModelCallDisposition.COMPLETED:
            if self.text is None or self.reason is not None:
                raise ValueError("A completed model response must contain text and no reason.")
        else:
            if self.reason is None:
                raise ValueError("An incomplete model response must explain its disposition.")
            if self.structured_output is not None:
                raise ValueError("An incomplete model response cannot contain structured output.")
        return self


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()
