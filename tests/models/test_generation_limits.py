"""Shared completion budget and reasoning-effort contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from yewo.models import MINIMUM_COMPLETION_TOKENS, ModelCallDisposition, ModelReasoningMode
from yewo.models.client import _RETRYABLE_RESPONSE_DISPOSITIONS
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models.profiles import ModelGenerationSettings
from yewo.models.records import ModelMessage, ModelMessageRole, ModelRequest, TextModelContent

from models.support import build_model_profile, build_test_client


def test_a_profile_cannot_request_fewer_than_the_shared_minimum_tokens() -> None:
    with pytest.raises(ValidationError):
        ModelGenerationSettings(max_completion_tokens=MINIMUM_COMPLETION_TOKENS - 1)


def test_a_profile_keeps_provider_reasoning_until_it_opts_out() -> None:
    default = ModelGenerationSettings(max_completion_tokens=MINIMUM_COMPLETION_TOKENS)

    assert default.reasoning is ModelReasoningMode.PROVIDER_DEFAULT
    disabled = ModelGenerationSettings(
        max_completion_tokens=MINIMUM_COMPLETION_TOKENS,
        reasoning=ModelReasoningMode.DISABLED,
    )
    assert disabled.reasoning is ModelReasoningMode.DISABLED


def test_an_empty_reply_is_retried_like_a_malformed_one() -> None:
    assert ModelCallDisposition.EMPTY in _RETRYABLE_RESPONSE_DISPOSITIONS
    assert ModelCallDisposition.MALFORMED in _RETRYABLE_RESPONSE_DISPOSITIONS
    assert ModelCallDisposition.TRUNCATED in _RETRYABLE_RESPONSE_DISPOSITIONS



def test_disabled_reasoning_reaches_each_provider_in_its_own_shape(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []

    class Recorded(Exception):
        """Stop the provider call once its arguments are known."""

    def complete(**kwargs: object) -> object:
        sent.append(kwargs)
        raise Recorded

    for provider, model in (
        ("anthropic", "anthropic/claude-sonnet-5"),
        ("openrouter", "openrouter/example/grounder"),
    ):
        sent.clear()
        profile = build_model_profile(provider=provider, model=model).model_copy(
            update={
                "generation": ModelGenerationSettings(
                    max_completion_tokens=MINIMUM_COMPLETION_TOKENS,
                    reasoning=ModelReasoningMode.DISABLED,
                )
            }
        )
        request = ModelRequest(
            request_id="request-1",
            profile_id=profile.profile_id,
            role=profile.role,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="hi"),),
                ),
            ),
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )
        root = tmp_path / provider
        root.mkdir()
        build_test_client(root, complete, profile).complete(request)
        arguments = sent[0]
        assert arguments["max_tokens"] == MINIMUM_COMPLETION_TOKENS
        if provider == "anthropic":
            assert arguments["thinking"] == {"type": "disabled"}
            assert "extra_body" not in arguments
        else:
            router_body = arguments["extra_body"]
            assert isinstance(router_body, dict)
            assert router_body["reasoning"] == {"enabled": False}
            assert "thinking" not in arguments
