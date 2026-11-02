"""Shared builders for model-client tests."""

from collections.abc import Callable
from pathlib import Path

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models.artifacts import ModelArtifactPolicy, ModelArtifactStore, ModelImageReader
from yewo.models.client import LiteLLMModelClient
from yewo.models.profiles import ModelProfileRegistry

from yewo.models import (
    ModelGenerationSettings,
    ModelImageConstraints,
    ModelMessageRole,
    ModelProfile,
    ModelRetryPolicy,
    ModelRole,
    StructuredOutputMode,
)


def build_model_profile(
    *,
    profile_id: str = "grounder-primary",
    role: ModelRole = ModelRole.GROUNDER,
    provider: str = "openrouter",
    model: str = "openrouter/example/grounder",
) -> ModelProfile:
    return ModelProfile(
        profile_id=profile_id,
        role=role,
        provider=provider,
        model=model,
        revision="revision-1",
        structured_output_mode=StructuredOutputMode.PROMPT_SCHEMA,
        generation=ModelGenerationSettings(max_completion_tokens=8_192, temperature=0),
        retry=ModelRetryPolicy(
            timeout_seconds=20,
            max_attempts=3,
            backoff_seconds=(1, 2),
        ),
        images=ModelImageConstraints(
            max_images=1,
            max_image_bytes=4_000_000,
            max_width_px=2048,
            max_height_px=2048,
        ),
        cache_control_role=(ModelMessageRole.SYSTEM if provider == "openrouter" else None),
        router_cache_affinity=provider == "openrouter",
        router_response_healing=provider == "openrouter",
    )


def build_test_client(
    tmp_path: Path,
    completion: Callable[..., object],
    profile: ModelProfile,
) -> LiteLLMModelClient:
    """Build one client around an explicit profile and a fake provider call."""
    identifiers = iter(f"artifact-{index}" for index in range(10))
    return LiteLLMModelClient(
        ModelProfileRegistry(profiles=(profile,)),
        ModelArtifactStore(
            tmp_path,
            ModelArtifactPolicy(
                sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
                retention=ArtifactRetention.RUN,
            ),
            lambda text: text,
            identifier_factory=lambda: next(identifiers),
        ),
        ModelImageReader(tmp_path),
        lambda text: text,
        completion=completion,
        calculate_cost=lambda _: None,
        credential_check=lambda _: (),
    )
