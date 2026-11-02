"""Role-specific model profile capacity contracts."""

import pytest
from pydantic import ValidationError

from yewo.models import ModelRole

from models.support import build_model_profile


def test_a_recovery_advisor_profile_must_accept_both_recovery_screenshots() -> None:
    with pytest.raises(ValidationError, match="at least 2 images"):
        build_model_profile(
            profile_id="recovery-primary",
            role=ModelRole.RECOVERY_ADVISOR,
            provider="anthropic",
            model="anthropic/claude-sonnet-5",
        )


def test_other_roles_keep_the_single_image_default() -> None:
    profile = build_model_profile(role=ModelRole.GROUNDER)

    assert profile.images.max_images == 1
