"""Normalize dynamic LiteLLM responses into typed model records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast

from litellm import exceptions
from pydantic import BaseModel, ValidationError

from yewo.models.records import ModelCallDisposition, ModelCallUsage


CostCalculator = Callable[[object], float | Decimal | None]

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_RETRYABLE_EXCEPTION_TYPES = (
    exceptions.Timeout,
    exceptions.RateLimitError,
    exceptions.APIConnectionError,
    exceptions.InternalServerError,
    exceptions.ServiceUnavailableError,
    ConnectionError,
    TimeoutError,
)


@dataclass(frozen=True)
class NormalizedModelCompletion:
    """Provider response fields accepted across the LiteLLM boundary."""

    disposition: ModelCallDisposition
    text: str | None
    structured_output: dict[str, object] | None
    reason: str | None
    provider_request_id: str | None
    served_model: str | None
    upstream_provider: str | None
    usage: ModelCallUsage
    provider_payload: dict[str, object]
    finish_reason: str | None = None
    validation_errors: tuple[dict[str, object], ...] = ()
    rejected_text: str | None = None


def normalize_litellm_completion(
    response: object,
    response_model: type[BaseModel] | None,
    calculate_cost: CostCalculator,
    validation_context: dict[str, object] | None = None,
) -> NormalizedModelCompletion:
    """Convert one untrusted LiteLLM response into a normalized completion."""
    payload = _mapping(response)
    hidden = _mapping(getattr(response, "_hidden_params", {}))
    usage = _normalize_usage(payload.get("usage"), response, calculate_cost)
    identity = {
        "provider_request_id": _optional_text(payload.get("id")),
        "served_model": _optional_text(payload.get("model")),
        "upstream_provider": _upstream_provider(payload, hidden),
        "usage": usage,
        "provider_payload": payload,
    }
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.MALFORMED,
            text=None,
            structured_output=None,
            reason="The provider response did not contain a completion choice.",
            **identity,
        )
    choice = _mapping(choices[0])
    finish_reason = _optional_text(choice.get("finish_reason"))
    message = _mapping(choice.get("message"))
    if _is_refusal(message, finish_reason):
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.REFUSED,
            text=None,
            structured_output=None,
            reason="The model refused the request.",
            finish_reason=finish_reason,
            **identity,
        )
    if finish_reason in {"length", "max_tokens"}:
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.TRUNCATED,
            text=None,
            structured_output=None,
            reason="The model response reached its output limit.",
            finish_reason=finish_reason,
            **identity,
        )
    content = message.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.EMPTY,
            text=None,
            structured_output=None,
            reason="The model returned no content.",
            finish_reason=finish_reason,
            **identity,
        )
    if not isinstance(content, str):
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.MALFORMED,
            text=None,
            structured_output=None,
            reason="The model returned content in an unsupported shape.",
            finish_reason=finish_reason,
            **identity,
        )
    structured_output, validation_errors = _validate_structured_output(
        content,
        response_model,
        validation_context,
    )
    if response_model is not None and structured_output is None:
        return NormalizedModelCompletion(
            disposition=ModelCallDisposition.MALFORMED,
            text=None,
            structured_output=None,
            reason="The model response did not match the required schema.",
            finish_reason=finish_reason,
            validation_errors=validation_errors,
            rejected_text=content,
            **identity,
        )
    return NormalizedModelCompletion(
        disposition=ModelCallDisposition.COMPLETED,
        text=content,
        structured_output=structured_output,
        reason=None,
        finish_reason=finish_reason,
        **identity,
    )


def provider_error_status(error: BaseException) -> int | None:
    """Return the integer HTTP status carried by a provider failure."""
    for field_name in ("status_code", "status"):
        value = getattr(error, field_name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def is_retryable_provider_error(error: BaseException) -> bool:
    """Return whether the same request could succeed after a delay."""
    return isinstance(error, _RETRYABLE_EXCEPTION_TYPES) or (
        provider_error_status(error) in _RETRYABLE_HTTP_STATUSES
    )


def describe_provider_error(error: BaseException) -> str:
    """Describe a provider failure without copying its message or prompt."""
    error_name = type(error).__name__
    status = provider_error_status(error)
    return f"{error_name} (HTTP {status})" if status is not None else error_name


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 - untrusted SDK response boundary
            return {}
        if isinstance(dumped, Mapping):
            return {str(key): item for key, item in dumped.items()}
    return {}


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _upstream_provider(
    payload: dict[str, object],
    hidden: dict[str, object],
) -> str | None:
    for source, field_name in (
        (payload, "upstream_provider"),
        (payload, "provider"),
        (payload, "provider_name"),
        (hidden, "upstream_provider"),
    ):
        provider = _optional_text(source.get(field_name))
        if provider is not None:
            return provider
    metadata = _mapping(payload.get("openrouter_metadata"))
    return _optional_text(metadata.get("provider_name")) or _optional_text(metadata.get("provider"))


def _is_refusal(message: dict[str, object], finish_reason: str | None) -> bool:
    refusal = message.get("refusal")
    return bool(refusal) or finish_reason in {"content_filter", "safety"}


def _validate_structured_output(
    content: str,
    response_model: type[BaseModel] | None,
    validation_context: dict[str, object] | None,
) -> tuple[dict[str, object] | None, tuple[dict[str, object], ...]]:
    if response_model is None:
        return None, ()
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        validated = response_model.model_validate_json(
            stripped,
            context=validation_context,
        )
    except ValidationError as error:
        errors = error.errors(include_url=False, include_input=False)
        return None, tuple(cast(dict[str, object], item) for item in errors)
    return cast(dict[str, object], validated.model_dump(mode="json")), ()


def _normalize_usage(
    value: object,
    response: object,
    calculate_cost: CostCalculator,
) -> ModelCallUsage:
    usage = _mapping(value)
    prompt_details = _mapping(usage.get("prompt_tokens_details"))
    completion_details = _mapping(usage.get("completion_tokens_details"))
    return ModelCallUsage(
        prompt_tokens=_optional_count(usage, "prompt_tokens"),
        completion_tokens=_optional_count(usage, "completion_tokens"),
        total_tokens=_optional_count(usage, "total_tokens"),
        cached_tokens=_optional_count(prompt_details, "cached_tokens"),
        cache_write_tokens=_first_optional_count(
            prompt_details,
            "cache_write_tokens",
            "cache_creation_tokens",
        ),
        reasoning_tokens=_optional_count(completion_details, "reasoning_tokens"),
        prompt_image_tokens=_optional_count(prompt_details, "image_tokens"),
        prompt_text_tokens=_optional_count(prompt_details, "text_tokens"),
        cost_usd=_resolve_cost(usage, response, calculate_cost),
        cost_details=_mapping(usage.get("cost_details")),
    )


def _optional_count(source: dict[str, object], field_name: str) -> int | None:
    value = source.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_optional_count(source: dict[str, object], *field_names: str) -> int | None:
    for field_name in field_names:
        count = _optional_count(source, field_name)
        if count is not None:
            return count
    return None


def _resolve_cost(
    usage: dict[str, object],
    response: object,
    calculate_cost: CostCalculator,
) -> Decimal | None:
    reported = _decimal(usage.get("cost"))
    if reported is not None:
        return reported
    try:
        return _decimal(calculate_cost(response))
    except Exception:  # noqa: BLE001 - optional telemetry at the SDK boundary
        return None


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return decimal if decimal.is_finite() and decimal >= 0 else None
