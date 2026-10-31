"""Provider-neutral model client and LiteLLM implementation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from typing import Protocol

import litellm
from litellm.cost_calculator import completion_cost
from pydantic import BaseModel

from yewo.models.artifacts import ModelArtifactStore, ModelImageReader, ModelResponseEvidence
from yewo.models.errors import ModelInputError
from yewo.models.message_encoding import ModelMessageEncoder
from yewo.models._litellm_response import (
    CostCalculator,
    NormalizedModelCompletion,
    describe_provider_error,
    is_retryable_provider_error,
    normalize_litellm_completion,
)
from yewo.models.profiles import (
    ModelProfile,
    ModelProfileRegistry,
    StructuredOutputMode,
    missing_model_credentials,
    validate_model_credentials,
)
from yewo.models.records import (
    ImageModelContent,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelResponse,
    TextModelContent,
    build_model_response_contract,
)

CompletionCall = Callable[..., object]
CredentialCheck = Callable[[ModelProfile], tuple[str, ...]]


class ModelClient(Protocol):
    """Complete provider-neutral model requests."""

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LiteLLMModelClient:
    """Call configured models only through the shared LiteLLM boundary."""

    def __init__(
        self,
        profiles: ModelProfileRegistry,
        artifacts: ModelArtifactStore,
        images: ModelImageReader,
        scrub_text: Callable[[str], str],
        *,
        completion: CompletionCall = litellm.completion,
        calculate_cost: CostCalculator = completion_cost,
        credential_check: CredentialCheck = missing_model_credentials,
        cache_affinity: str | None = None,
        wait: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        time_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        validate_model_credentials(profiles, credential_check)
        self._profiles = profiles
        self._artifacts = artifacts
        self._images = images
        self._scrub_text = scrub_text
        self._completion = completion
        self._calculate_cost = calculate_cost
        self._wait = wait
        self._monotonic = monotonic
        self._time_factory = time_factory
        self._cache_affinity = (
            f"yewo-{sha256(cache_affinity.encode('utf-8')).hexdigest()}" if cache_affinity else None
        )

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        """Complete one request with bounded provider and validation attempts."""
        profile = self._profiles.resolve(request.profile_id, request.role)
        self._validate_response_contract(request, response_model)
        base_messages = self._prepare_messages(request.messages, profile)
        response_format = self._response_format(profile, request)
        calls: list[ModelCallRecord] = []
        corrections: tuple[ModelMessage, ...] = ()
        message_encoder = ModelMessageEncoder(self._images, profile)
        validation_attempts = (
            profile.retry.max_validation_attempts if response_model is not None else 1
        )
        actual_attempt = 0
        for validation_attempt in range(1, validation_attempts + 1):
            messages = self._attempt_messages(
                base_messages,
                request,
                profile.structured_output_mode,
                corrections,
            )
            provider_messages = message_encoder.provider_messages(messages)
            artifact_messages = message_encoder.artifact_messages(messages)
            completion_arguments = self._completion_arguments(
                profile,
                provider_messages,
                response_format,
            )
            prompt_artifact = self._artifacts.persist_request(
                request,
                artifact_messages,
                response_format,
                completion_arguments,
            )
            for provider_attempt in range(1, profile.retry.max_attempts + 1):
                actual_attempt += 1
                started_at = self._time_factory()
                started = self._monotonic()
                try:
                    provider_response = self._completion(**completion_arguments)
                except Exception as error:  # noqa: BLE001 - provider boundary
                    latency_ms = max(0, (self._monotonic() - started) * 1000)
                    reason = describe_provider_error(error)
                    response_evidence = self._artifacts.prepare_response(
                        request_id=request.request_id,
                        attempt_number=actual_attempt,
                        disposition=ModelCallDisposition.PROVIDER_ERROR,
                        text=None,
                        reason=reason,
                    )
                    call = self._provider_error_call(
                        request,
                        profile,
                        actual_attempt,
                        validation_attempt,
                        provider_attempt,
                        started_at,
                        latency_ms,
                        prompt_artifact,
                    )
                    calls.append(self._persist_call(call, response_evidence))
                    if (
                        not is_retryable_provider_error(error)
                        or provider_attempt == profile.retry.max_attempts
                    ):
                        return self._response(
                            request,
                            ModelCallDisposition.PROVIDER_ERROR,
                            tuple(calls),
                            reason=reason,
                        )
                    self._wait(profile.retry.backoff_seconds[provider_attempt - 1])
                    continue
                latency_ms = max(0, (self._monotonic() - started) * 1000)
                normalized = normalize_litellm_completion(
                    provider_response,
                    response_model,
                    self._calculate_cost,
                    validation_context,
                )
                response_evidence = self._artifacts.prepare_response(
                    request_id=request.request_id,
                    attempt_number=actual_attempt,
                    disposition=normalized.disposition,
                    text=normalized.text,
                    reason=normalized.reason,
                    provider_payload=normalized.provider_payload,
                )
                call = self._completed_call(
                    request,
                    profile,
                    actual_attempt,
                    validation_attempt,
                    provider_attempt,
                    started_at,
                    latency_ms,
                    prompt_artifact,
                    normalized,
                )
                calls.append(self._persist_call(call, response_evidence))
                if (
                    response_model is not None
                    and normalized.disposition is ModelCallDisposition.MALFORMED
                    and validation_attempt < validation_attempts
                ):
                    corrections = self._correction_messages(normalized)
                    break
                return self._response(
                    request,
                    normalized.disposition,
                    tuple(calls),
                    text=normalized.text,
                    structured_output=normalized.structured_output,
                    reason=normalized.reason,
                )
        raise RuntimeError("The model validation policy allowed no final response.")

    @staticmethod
    def _validate_response_contract(
        request: ModelRequest,
        response_model: type[BaseModel] | None,
    ) -> None:
        if request.response_contract is None and response_model is None:
            return
        if request.response_contract is None or response_model is None:
            raise ModelInputError("A structured model request requires its local response model.")
        expected = build_model_response_contract(
            request.response_contract.name,
            response_model.model_json_schema(),
        )
        if expected.schema_sha256 != request.response_contract.schema_sha256:
            raise ModelInputError("The local response model does not match the request contract.")

    def _prepare_messages(
        self,
        messages: tuple[ModelMessage, ...],
        profile: ModelProfile,
    ) -> tuple[ModelMessage, ...]:
        image_count = sum(
            isinstance(content, ImageModelContent)
            for message in messages
            for content in message.content
        )
        if image_count > profile.images.max_images:
            raise ModelInputError("The model request exceeds the profile image-count limit.")
        return tuple(
            message.model_copy(
                update={
                    "content": tuple(self._prepare_content(content) for content in message.content)
                }
            )
            for message in messages
        )

    def _prepare_content(
        self,
        content: TextModelContent | ImageModelContent,
    ) -> TextModelContent | ImageModelContent:
        if isinstance(content, ImageModelContent):
            return content
        return TextModelContent(text=self._scrub_input_text(content.text))

    def _scrub_input_text(self, text: str) -> str:
        try:
            scrubbed = self._scrub_text(text)
        except Exception as error:  # noqa: BLE001 - injected security boundary
            raise ModelInputError("Could not scrub model input text.") from error
        if not isinstance(scrubbed, str) or not scrubbed:
            raise ModelInputError("The model input scrubber returned an invalid value.")
        return scrubbed

    @staticmethod
    def _attempt_messages(
        base_messages: tuple[ModelMessage, ...],
        request: ModelRequest,
        mode: StructuredOutputMode,
        corrections: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        messages = list(base_messages)
        messages.extend(corrections)
        contract = request.response_contract
        if contract is None or mode is StructuredOutputMode.NATIVE_SCHEMA:
            return tuple(messages)
        schema = json.dumps(contract.json_schema, sort_keys=True, separators=(",", ":"))
        messages.append(
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(
                    TextModelContent(
                        text=(
                            "Return one JSON object that matches this schema. "
                            f"Do not include other text.\n{schema}"
                        )
                    ),
                ),
            )
        )
        return tuple(messages)

    @staticmethod
    def _response_format(
        profile: ModelProfile,
        request: ModelRequest,
    ) -> dict[str, object] | None:
        if (
            request.response_contract is None
            or profile.structured_output_mode is not StructuredOutputMode.NATIVE_SCHEMA
        ):
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_contract.name,
                "schema": request.response_contract.json_schema,
                "strict": True,
            },
        }

    def _completion_arguments(
        self,
        profile: ModelProfile,
        messages: list[dict[str, object]],
        response_format: dict[str, object] | None,
    ) -> dict[str, object]:
        arguments: dict[str, object] = {
            "model": profile.model,
            "messages": messages,
            "timeout": profile.retry.timeout_seconds,
            "max_completion_tokens": profile.generation.max_completion_tokens,
        }
        optional_generation_settings = {
            "temperature": profile.generation.temperature,
            "top_p": profile.generation.top_p,
            "seed": profile.generation.seed,
        }
        for field_name, value in optional_generation_settings.items():
            if value is not None:
                arguments[field_name] = value
        if profile.endpoint is not None:
            arguments["base_url"] = profile.endpoint.base_url
        if response_format is not None:
            arguments["response_format"] = response_format
        if profile.cache_control_role is not None:
            arguments["cache_control_injection_points"] = [
                {"location": "message", "role": profile.cache_control_role.value}
            ]
        router_options: dict[str, object] = {}
        if profile.router_response_healing:
            router_options["plugins"] = [{"id": "response-healing"}]
        if profile.router_cache_affinity and self._cache_affinity is not None:
            router_options.update(
                {
                    "session_id": self._cache_affinity,
                    "prompt_cache_key": self._cache_affinity,
                }
            )
        if router_options:
            arguments["extra_body"] = router_options
        return arguments

    def _correction_messages(
        self,
        normalized: NormalizedModelCompletion,
    ) -> tuple[ModelMessage, ...]:
        if normalized.rejected_text is None:
            raise ModelInputError("A malformed structured response has no rejected text.")
        rejected_text = self._scrub_input_text(normalized.rejected_text)
        errors = json.dumps(normalized.validation_errors, default=str, separators=(",", ":"))
        text = self._scrub_input_text(
            "Your previous reply did not match the required JSON schema:\n"
            f"{errors}\n"
            "Answer again as JSON matching the schema exactly."
        )
        return (
            ModelMessage(
                role=ModelMessageRole.ASSISTANT,
                content=(TextModelContent(text=rejected_text),),
            ),
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(TextModelContent(text=text),),
            ),
        )

    def _persist_call(
        self,
        call: ModelCallRecord,
        response_evidence: ModelResponseEvidence,
    ) -> ModelCallRecord:
        reference = self._artifacts.persist_call(call, response_evidence)
        return call.model_copy(update={"call_artifact": reference})

    @staticmethod
    def _response(
        request: ModelRequest,
        disposition: ModelCallDisposition,
        calls: tuple[ModelCallRecord, ...],
        *,
        text: str | None = None,
        structured_output: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> ModelResponse:
        return ModelResponse(
            request_id=request.request_id,
            disposition=disposition,
            text=text,
            structured_output=structured_output,
            reason=reason,
            calls=calls,
            sensitivity=request.sensitivity,
            retention=request.retention,
        )

    @staticmethod
    def _provider_error_call(
        request: ModelRequest,
        profile: ModelProfile,
        attempt_number: int,
        validation_attempt_number: int,
        provider_attempt_number: int,
        started_at: datetime,
        latency_ms: float,
        prompt_artifact,
    ) -> ModelCallRecord:
        return ModelCallRecord(
            request_id=request.request_id,
            profile_id=profile.profile_id,
            role=profile.role,
            requested_provider=profile.provider,
            requested_model=profile.model,
            requested_revision=profile.revision,
            attempt_number=attempt_number,
            validation_attempt_number=validation_attempt_number,
            provider_attempt_number=provider_attempt_number,
            started_at=started_at,
            latency_ms=latency_ms,
            disposition=ModelCallDisposition.PROVIDER_ERROR,
            usage=ModelCallUsage(),
            prompt_artifact=prompt_artifact,
        )

    @staticmethod
    def _completed_call(
        request: ModelRequest,
        profile: ModelProfile,
        attempt_number: int,
        validation_attempt_number: int,
        provider_attempt_number: int,
        started_at: datetime,
        latency_ms: float,
        prompt_artifact,
        normalized: NormalizedModelCompletion,
    ) -> ModelCallRecord:
        return ModelCallRecord(
            request_id=request.request_id,
            profile_id=profile.profile_id,
            role=profile.role,
            requested_provider=profile.provider,
            requested_model=profile.model,
            requested_revision=profile.revision,
            served_model=normalized.served_model,
            upstream_provider=normalized.upstream_provider,
            provider_request_id=normalized.provider_request_id,
            attempt_number=attempt_number,
            validation_attempt_number=validation_attempt_number,
            provider_attempt_number=provider_attempt_number,
            started_at=started_at,
            latency_ms=latency_ms,
            finish_reason=normalized.finish_reason,
            disposition=normalized.disposition,
            usage=normalized.usage,
            prompt_artifact=prompt_artifact,
        )
