"""Validated configuration for provider-neutral model selection."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from enum import StrEnum
import io
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from litellm.utils import validate_environment

from yewo.artifacts import RecordIdentifier
from yewo.models.records import ModelMessageRole, ModelRole


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
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError("The model endpoint URL has an invalid port.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (":" in parsed.netloc and port is None)
        ):
            raise ValueError("The model endpoint must be an HTTP URL without credentials.")
        return value.rstrip("/")


class ModelGenerationSettings(BaseModel):
    """Explicit provider-neutral generation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_completion_tokens: int = Field(gt=0)
    seed: int | None = None


class ModelRetryPolicy(BaseModel):
    """Bound one logical request and its transient retries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(default=1, ge=1, le=5)
    backoff_seconds: tuple[float, ...] = ()
    max_validation_attempts: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def _validate_backoff(self) -> Self:
        if len(self.backoff_seconds) != self.max_attempts - 1:
            raise ValueError("Retry backoff must contain one delay between each attempt.")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError("Retry backoff delays cannot be negative.")
        return self


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
    cache_control_role: ModelMessageRole | None = ModelMessageRole.SYSTEM
    router_cache_affinity: bool = False
    router_response_healing: bool = False

    @model_validator(mode="after")
    def _validate_router_settings(self) -> Self:
        if (
            self.router_cache_affinity or self.router_response_healing
        ) and self.provider != "openrouter":
            raise ValueError("Router options require an OpenRouter profile.")
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


def missing_model_credentials(profile: ModelProfile) -> tuple[str, ...]:
    """Return credential names LiteLLM requires for one profile."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = validate_environment(profile.model)
    missing = result.get("missing_keys") if isinstance(result, dict) else None
    return tuple(str(name) for name in missing or ())
