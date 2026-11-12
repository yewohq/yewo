"""Tests for provider-neutral model records and profiles."""

from datetime import datetime, timezone
import os

import pytest
from pydantic import ValidationError

from yewo.artifacts import ArtifactRetention
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelCallRecord,
    ModelEndpoint,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelResponse,
    ModelProfileRegistry,
    ModelRetryPolicy,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)
from yewo.models.profiles import missing_model_credentials
from models.support import build_model_profile
from observations.factories import build_screenshot


def _call(
    disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
) -> ModelCallRecord:
    return ModelCallRecord(
        request_id="request-1",
        profile_id="grounder-primary",
        role=ModelRole.GROUNDER,
        requested_provider="openrouter",
        requested_model="openrouter/example/grounder",
        requested_revision="revision-1",
        served_model="example/grounder",
        upstream_provider="example-provider",
        provider_request_id="provider-call-1",
        attempt_number=1,
        started_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
        latency_ms=125,
        disposition=disposition,
    )


def test_model_request_serialization_preserves_schema_and_image_identity() -> None:
    contract = build_model_response_contract(
        "point-response",
        {"type": "object", "properties": {"x": {"type": "number"}}},
    )
    request = ModelRequest(
        request_id="request-1",
        profile_id="grounder-primary",
        role=ModelRole.GROUNDER,
        messages=(
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(
                    TextModelContent(text="Select the save button."),
                    ImageModelContent(screenshot=build_screenshot()),
                ),
            ),
        ),
        response_contract=contract,
        retention=ArtifactRetention.RUN,
    )

    restored = ModelRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert restored.response_contract is not None
    assert restored.response_contract.schema_sha256 == contract.schema_sha256


def test_response_contract_rejects_a_changed_schema() -> None:
    contract = build_model_response_contract("answer", {"type": "object"})
    payload = contract.model_dump()
    payload["json_schema"] = {"type": "array"}

    with pytest.raises(ValidationError, match="digest"):
        type(contract).model_validate(payload)


@pytest.mark.parametrize(
    ("disposition", "text", "reason"),
    [
        (ModelCallDisposition.EMPTY, None, "The provider returned no content."),
        (ModelCallDisposition.REFUSED, None, "The model refused the request."),
        (ModelCallDisposition.TRUNCATED, None, "The output reached its limit."),
        (ModelCallDisposition.MALFORMED, None, "The output did not match the schema."),
        (ModelCallDisposition.PROVIDER_ERROR, None, "The provider did not answer."),
    ],
)
def test_incomplete_response_dispositions_remain_distinct(
    disposition: ModelCallDisposition,
    text: str | None,
    reason: str,
) -> None:
    response = ModelResponse(
        request_id="request-1",
        disposition=disposition,
        text=text,
        reason=reason,
        calls=(_call(disposition),),
        retention=ArtifactRetention.RUN,
    )

    assert response.disposition is disposition
    assert response.reason == reason


def test_completed_response_requires_text_and_matching_call_identity() -> None:
    with pytest.raises(ValidationError, match="contain text"):
        ModelResponse(
            request_id="request-1",
            disposition=ModelCallDisposition.COMPLETED,
            calls=(_call(),),
            retention=ArtifactRetention.RUN,
        )

    mismatched_call = _call().model_copy(update={"request_id": "request-2"})
    with pytest.raises(ValidationError, match="match the response request"):
        ModelResponse(
            request_id="request-1",
            disposition=ModelCallDisposition.COMPLETED,
            text="done",
            calls=(mismatched_call,),
            retention=ArtifactRetention.RUN,
        )


def test_model_call_rejects_unsafe_artifact_paths_and_naive_time() -> None:
    payload = _call().model_dump()
    payload["started_at"] = datetime(2026, 10, 30)
    with pytest.raises(ValidationError, match="include a timezone"):
        ModelCallRecord.model_validate(payload)


def test_registry_resolves_exact_profile_without_fallback() -> None:
    primary = build_model_profile()
    secondary = build_model_profile(
        profile_id="grounder-secondary",
        model="openrouter/example/second-grounder",
    )
    registry = ModelProfileRegistry(profiles=(primary, secondary))

    assert registry.resolve("grounder-secondary", ModelRole.GROUNDER) == secondary
    with pytest.raises(LookupError, match="not configured"):
        registry.resolve("missing", ModelRole.GROUNDER)
    with pytest.raises(LookupError, match="cannot serve"):
        registry.resolve("grounder-primary", ModelRole.TASK_PLANNER)


def test_registry_rejects_duplicate_profile_identifiers() -> None:
    profile = build_model_profile()

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        ModelProfileRegistry(profiles=(profile, profile))


def test_endpoint_rejects_embedded_credentials_and_normalizes_trailing_slash() -> None:
    with pytest.raises(ValidationError, match="without credentials"):
        ModelEndpoint(base_url="https://secret@example.test/v1")

    endpoint = ModelEndpoint(base_url="https://models.example.test/v1/")
    assert endpoint.base_url == "https://models.example.test/v1"


def test_retry_policy_requires_one_delay_between_attempts() -> None:
    with pytest.raises(ValidationError, match="one delay"):
        ModelRetryPolicy(timeout_seconds=10, max_attempts=3, backoff_seconds=(1,))


def test_router_options_require_an_explicit_openrouter_profile() -> None:
    payload = build_model_profile(
        provider="anthropic",
        model="anthropic/example",
    ).model_dump()
    payload["router_cache_affinity"] = True

    with pytest.raises(ValidationError, match="OpenRouter profile"):
        type(build_model_profile()).model_validate(payload)


def test_credential_preflight_uses_litellm_without_printing_provider_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def validate_environment(model: str) -> dict[str, object]:
        print(f"provider banner for {model}")
        return {"missing_keys": ["EXAMPLE_API_KEY"]}

    monkeypatch.setattr(
        "yewo.models.profiles.validate_environment",
        validate_environment,
    )

    assert missing_model_credentials(build_model_profile()) == ("EXAMPLE_API_KEY",)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("credential", ("", "   "))
def test_credential_preflight_treats_a_blank_key_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    credential: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", credential)

    assert missing_model_credentials(build_model_profile()) == ("OPENROUTER_API_KEY",)
    assert os.environ["OPENROUTER_API_KEY"] == credential
