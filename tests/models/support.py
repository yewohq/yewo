"""Shared builders for model-client tests."""

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
        generation=ModelGenerationSettings(max_completion_tokens=128, temperature=0),
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
