"""Validated configuration for provider-neutral model selection."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from enum import StrEnum
import io
import os
from threading import Lock
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from litellm.utils import validate_environment

from yewo.artifacts import RecordIdentifier
from yewo.models.errors import ModelInputError
from yewo.models.records import ModelMessageRole, ModelRole


MINIMUM_COMPLETION_TOKENS = 8_192


class StructuredOutputMode(StrEnum):
    """How a profile asks a model for locally validated JSON."""

    PROMPT_SCHEMA = "prompt_schema"
    NATIVE_SCHEMA = "native_schema"


class ModelEndpoint(BaseModel):
    """A provider endpoint without credential values."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    base_url: str = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        try:
            url_parts = urlsplit(value)
            port = url_parts.port
        except ValueError as error:
            raise ValueError("The model endpoint URL has an invalid port.") from error
        if (
            url_parts.scheme not in {"http", "https"}
            or not url_parts.hostname
            or url_parts.username is not None
            or url_parts.password is not None
            or url_parts.query
            or url_parts.fragment
            or (":" in url_parts.netloc and port is None)
        ):
            raise ValueError("The model endpoint must be an HTTP URL without credentials.")
        return value.rstrip("/")


class ModelReasoningMode(StrEnum):
    """Reasoning control a provider accepts through the LiteLLM boundary.

    Some providers reject an explicit thinking budget, so a profile can only
    keep the provider default or turn reasoning off.
    """

    PROVIDER_DEFAULT = "provider_default"
    DISABLED = "disabled"


class ModelGenerationSettings(BaseModel):
    """Explicit provider-neutral generation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_completion_tokens: int = Field(ge=MINIMUM_COMPLETION_TOKENS)
    reasoning: ModelReasoningMode = ModelReasoningMode.PROVIDER_DEFAULT
    seed: int | None = None


class ModelRetryPolicy(BaseModel):
    """Bound one logical request and its transient retries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(default=4, ge=1, le=5)
    backoff_seconds: tuple[float, ...] = (1, 2, 4)
    max_validation_attempts: int = Field(default=4, ge=1, le=4)
    maximum_retry_delay_seconds: float = Field(default=300, gt=0, le=900)

    @model_validator(mode="after")
    def _validate_backoff(self) -> Self:
        if len(self.backoff_seconds) != self.max_attempts - 1:
            raise ValueError("Retry backoff must contain one delay between each attempt.")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError("Retry backoff delays cannot be negative.")
        return self


_MINIMUM_ROLE_IMAGES = {
    ModelRole.RECOVERY_ADVISOR: 2,
    ModelRole.SCREEN_ANALYZER: 1,
    ModelRole.TASK_PLANNER: 2,
    ModelRole.TEST_DESIGNER: 2,
}
_CREDENTIAL_ENVIRONMENT_LOCK = Lock()


def minimum_image_count_for_role(role: ModelRole) -> int:
    """Return the image capacity required by one model role."""
    return _MINIMUM_ROLE_IMAGES.get(role, 1)


class ModelImageConstraints(BaseModel):
    """Limits applied before image content reaches a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_images: int = Field(default=1, ge=0)
    max_image_bytes: int = Field(gt=0)
    max_width_px: int = Field(gt=0)
    max_height_px: int = Field(gt=0)


class ModelProfile(BaseModel):
    """One reproducible model configuration for a product role."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    profile_id: RecordIdentifier
    role: ModelRole
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    endpoint: ModelEndpoint | None = None
    structured_output_mode: StructuredOutputMode
    generation: ModelGenerationSettings
    retry: ModelRetryPolicy
    images: ModelImageConstraints
    cache_control_role: ModelMessageRole | None = None
    router_cache_affinity: bool = False
    router_response_healing: bool = False

    @model_validator(mode="after")
    def _validate_router_settings(self) -> Self:
        if (
            self.router_cache_affinity or self.router_response_healing
        ) and self.provider != "openrouter":
            raise ValueError("Router options require an OpenRouter profile.")
        return self

    @model_validator(mode="after")
    def _validate_role_image_capacity(self) -> Self:
        required = _MINIMUM_ROLE_IMAGES.get(self.role)
        if required is not None and self.images.max_images < required:
            raise ValueError(f"A {self.role.value} profile must accept at least {required} images.")
        return self


class ModelProfileRegistry(BaseModel):
    """Resolve immutable profiles without fallback or role substitution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profiles: tuple[ModelProfile, ...]

    @model_validator(mode="after")
    def _validate_profile_ids(self) -> Self:
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Model profile identifiers must be unique.")
        return self

    def resolve(self, profile_id: str, role: ModelRole) -> ModelProfile:
        """Return the exact selected profile with the required role."""
        matching = tuple(profile for profile in self.profiles if profile.profile_id == profile_id)
        if not matching:
            raise LookupError(f"The model profile is not configured: {profile_id}")
        profile = matching[0]
        if profile.role is not role:
            raise LookupError(f"The model profile {profile_id} cannot serve the {role.value} role.")
        return profile


def build_model_selections(profiles: ModelProfileRegistry) -> dict[str, str]:
    """Return the model identifier configured for each role.

    A role with several configured profiles lists each model once.
    """
    models_by_role: dict[str, set[str]] = {}
    for profile in profiles.profiles:
        role_models = models_by_role.setdefault(profile.role.value, set())
        role_models.add(profile.model)
    return {role: ", ".join(sorted(models)) for role, models in models_by_role.items()}


def missing_model_credentials(profile: ModelProfile) -> tuple[str, ...]:
    """Return credential names LiteLLM requires for one profile."""
    with _CREDENTIAL_ENVIRONMENT_LOCK:
        # LiteLLM treats key membership as sufficient. Hide blank values during this startup probe.
        blank_values = {name: value for name, value in os.environ.items() if not value.strip()}
        try:
            for name in blank_values:
                del os.environ[name]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = validate_environment(profile.model)
        finally:
            os.environ.update(blank_values)
    missing_keys = result.get("missing_keys") if isinstance(result, dict) else None
    return tuple(str(name) for name in missing_keys or ())


def validate_model_credentials(
    profiles: ModelProfileRegistry,
    credential_check=missing_model_credentials,
) -> None:
    """Fail before device work when any configured profile lacks credentials."""
    for profile in profiles.profiles:
        try:
            missing_keys = credential_check(profile)
        except Exception as error:  # noqa: BLE001 - LiteLLM configuration boundary
            raise ModelInputError("Could not validate model credentials.") from error
        if missing_keys:
            names = ", ".join(missing_keys)
            raise ModelInputError(f"Missing credentials for profile {profile.profile_id}: {names}.")
