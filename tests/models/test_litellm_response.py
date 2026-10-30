"""Tests for normalization of untrusted LiteLLM response envelopes."""

from decimal import Decimal

from pydantic import BaseModel, ValidationInfo, field_validator
import pytest

from yewo.models import ModelCallDisposition
from yewo.models._litellm_response import (
    describe_provider_error,
    is_retryable_provider_error,
    normalize_litellm_completion,
    provider_error_status,
)


class StructuredAnswer(BaseModel):
    target: str
    confidence: float


class ContextAnswer(BaseModel):
    target: str

    @field_validator("target")
    @classmethod
    def _validate_target(cls, value: str, info: ValidationInfo) -> str:
        if value != (info.context or {}).get("required_target"):
            raise ValueError("The target does not match the validation context.")
        return value


def _response(
    content: object = "ok",
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": "provider-call-1",
        "model": "served-model-1",
        "upstream_provider": "provider-a",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "refusal": refusal},
            }
        ],
        "usage": usage or {},
    }


@pytest.mark.parametrize(
    ("response", "disposition"),
    [
        (_response(""), ModelCallDisposition.EMPTY),
        (_response("partial", finish_reason="length"), ModelCallDisposition.TRUNCATED),
        (_response(None, refusal="blocked"), ModelCallDisposition.REFUSED),
        ({"choices": []}, ModelCallDisposition.MALFORMED),
        (_response(["unexpected"]), ModelCallDisposition.MALFORMED),
    ],
)
def test_response_failures_remain_distinct(
    response: dict[str, object],
    disposition: ModelCallDisposition,
) -> None:
    normalized = normalize_litellm_completion(response, None, lambda _: None)

    assert normalized.disposition is disposition
    assert normalized.text is None
    assert normalized.reason is not None


def test_structured_response_is_validated_locally() -> None:
    normalized = normalize_litellm_completion(
        _response('{"target":"Save","confidence":0.9}'),
        StructuredAnswer,
        lambda _: None,
    )

    assert normalized.disposition is ModelCallDisposition.COMPLETED
    assert normalized.structured_output == {"target": "Save", "confidence": 0.9}

    malformed = normalize_litellm_completion(
        _response('{"target":"Save"}'),
        StructuredAnswer,
        lambda _: None,
    )
    assert malformed.disposition is ModelCallDisposition.MALFORMED


def test_usage_normalization_preserves_attribution_and_unknown_cost() -> None:
    normalized = normalize_litellm_completion(
        _response(
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {
                    "cached_tokens": 4,
                    "cache_creation_tokens": 2,
                    "image_tokens": 6,
                },
                "completion_tokens_details": {"reasoning_tokens": 1},
                "cost_details": {"provider": 0.01},
            }
        ),
        None,
        lambda _: None,
    )

    assert normalized.provider_request_id == "provider-call-1"
    assert normalized.served_model == "served-model-1"
    assert normalized.upstream_provider == "provider-a"
    assert normalized.usage.total_tokens == 13
    assert normalized.usage.cached_tokens == 4
    assert normalized.usage.cache_write_tokens == 2
    assert normalized.usage.prompt_image_tokens == 6
    assert normalized.usage.cost_usd is None
    assert normalized.usage.cost_details == {"provider": 0.01}
    assert normalized.finish_reason == "stop"


def test_reported_cost_precedes_calculated_cost() -> None:
    normalized = normalize_litellm_completion(
        _response(usage={"cost": 0.0123}),
        None,
        lambda _: Decimal("9.99"),
    )

    assert normalized.usage.cost_usd == Decimal("0.0123")


def test_structured_response_accepts_json_fences_and_validation_context() -> None:
    normalized = normalize_litellm_completion(
        _response('```json\n{"target":"Save"}\n```'),
        ContextAnswer,
        lambda _: None,
        {"required_target": "Save"},
    )

    assert normalized.disposition is ModelCallDisposition.COMPLETED
    assert normalized.structured_output == {"target": "Save"}


class ProviderFailure(RuntimeError):
    def __init__(self, status_code: int, message: str = "sensitive provider message") -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500, 502, 503, 504, 529])
def test_retryable_provider_statuses_are_explicit(status_code: int) -> None:
    assert is_retryable_provider_error(ProviderFailure(status_code))


def test_terminal_provider_error_is_not_retried_or_quoted() -> None:
    error = ProviderFailure(401, "prompt and credential appeared here")

    assert not is_retryable_provider_error(error)
    assert provider_error_status(error) == 401
    assert describe_provider_error(error) == "ProviderFailure (HTTP 401)"


def test_transport_failures_without_status_are_retryable() -> None:
    assert is_retryable_provider_error(TimeoutError("request body"))
    assert describe_provider_error(TimeoutError("request body")) == "TimeoutError"
