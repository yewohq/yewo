"""Offline integration tests for the LiteLLM model client."""

from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from pydantic import BaseModel
import pytest

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import (
    ImageModelContent,
    LiteLLMModelClient,
    ModelArtifactPolicy,
    ModelArtifactStore,
    ModelCallDisposition,
    ModelImageReader,
    ModelInputError,
    ModelMessage,
    ModelMessageRole,
    ModelProfileRegistry,
    ModelRequest,
    ModelRole,
    StructuredOutputMode,
    TextModelContent,
    build_model_response_contract,
)
from yewo.observations import RedactionProvenance, ScreenshotStore
from models.support import build_model_profile


def _artifact_payload(root: Path, reference) -> dict[str, object]:
    return next(
        payload
        for payload in (
            json.loads(line) for line in (root / reference.path).read_text().splitlines()
        )
        if payload["artifact_id"] == reference.artifact_id
    )


class StructuredAnswer(BaseModel):
    target: str
    confidence: float


class CountingImageReader(ModelImageReader):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.read_count = 0

    def read(self, reference):
        self.read_count += 1
        return super().read(reference)


def _png() -> bytes:
    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _screenshot(tmp_path: Path):
    return ScreenshotStore(tmp_path, identifier_factory=lambda: "shot-1").persist_redacted(
        _png(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
        redaction=RedactionProvenance(
            policy_id="region-mask",
            region_count=0,
            region_set_sha256="c" * 64,
            input_sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        ),
    )


def _request(tmp_path: Path, *, structured: bool = False, image: bool = False) -> ModelRequest:
    content: list[TextModelContent | ImageModelContent] = [TextModelContent(text="Select Save.")]
    if image:
        content.append(ImageModelContent(screenshot=_screenshot(tmp_path)))
    return ModelRequest(
        request_id="request-1",
        profile_id="grounder-primary",
        role=ModelRole.GROUNDER,
        messages=(ModelMessage(role=ModelMessageRole.USER, content=tuple(content)),),
        response_contract=(
            build_model_response_contract(
                "structured-answer",
                StructuredAnswer.model_json_schema(),
            )
            if structured
            else None
        ),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def _response(
    content: object = "ok",
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> dict[str, object]:
    return {
        "id": "provider-call-1",
        "model": "example/grounder-revision-1",
        "upstream_provider": "example-provider",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _client(
    tmp_path: Path,
    completion: Callable[..., object],
    *,
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.PROMPT_SCHEMA,
    waits: list[float] | None = None,
    scrub_text: Callable[[str], str] = lambda text: text,
    credential_check: Callable[..., tuple[str, ...]] = lambda _: (),
    image_reader: ModelImageReader | None = None,
) -> LiteLLMModelClient:
    profile = build_model_profile().model_copy(
        update={"structured_output_mode": structured_output_mode}
    )
    identifiers = iter(f"artifact-{index}" for index in range(10))
    moments = iter(float(index) for index in range(20))
    return LiteLLMModelClient(
        ModelProfileRegistry(profiles=(profile,)),
        ModelArtifactStore(
            tmp_path,
            ModelArtifactPolicy(
                sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
                retention=ArtifactRetention.RUN,
            ),
            scrub_text,
            identifier_factory=lambda: next(identifiers),
        ),
        image_reader or ModelImageReader(tmp_path),
        scrub_text,
        completion=completion,
        calculate_cost=lambda _: None,
        credential_check=credential_check,
        cache_affinity="run-1",
        wait=(waits.append if waits is not None else lambda _: None),
        monotonic=lambda: next(moments),
        time_factory=lambda: datetime(2026, 10, 30, tzinfo=timezone.utc),
    )


def test_client_sends_explicit_settings_and_records_attribution(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response()

    response = _client(tmp_path, complete).complete(_request(tmp_path))

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert calls[0]["model"] == "openrouter/example/grounder"
    assert calls[0]["timeout"] == 20
    assert calls[0]["max_completion_tokens"] == 128
    assert calls[0]["temperature"] == 0
    assert calls[0]["cache_control_injection_points"] == [{"location": "message", "role": "system"}]
    extra_body = calls[0]["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["plugins"] == [{"id": "response-healing"}]
    assert extra_body["session_id"] == extra_body["prompt_cache_key"]
    assert str(extra_body["session_id"]).startswith("yewo-")
    assert "run-1" not in str(extra_body)
    assert response.calls[0].role is ModelRole.GROUNDER
    assert response.calls[0].served_model == "example/grounder-revision-1"
    assert response.calls[0].upstream_provider == "example-provider"
    assert response.calls[0].usage.cost_usd is None
    assert response.calls[0].prompt_artifact is not None
    assert response.calls[0].call_artifact is not None
    call_payload = _artifact_payload(tmp_path, response.calls[0].call_artifact)
    assert call_payload["requested_model"] == "openrouter/example/grounder"
    assert call_payload["finish_reason"] == "stop"
    usage_payload = call_payload["usage"]
    assert isinstance(usage_payload, dict)
    assert usage_payload["total_tokens"] == 12


def test_client_scrubs_declared_values_before_provider_call(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response()

    request = _request(tmp_path).model_copy(
        update={
            "messages": (
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="Select hunter2 account."),),
                ),
            )
        }
    )
    _client(
        tmp_path,
        complete,
        scrub_text=lambda text: text.replace("hunter2", "{{password}}"),
    ).complete(request)

    assert "hunter2" not in str(calls[0]["messages"])
    assert "{{password}}" in str(calls[0]["messages"])


def test_prompt_schema_mode_validates_locally_without_provider_grammar(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response('{"target":"Save","confidence":0.9}')

    response = _client(tmp_path, complete).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    assert "structured-answer" not in str(messages)
    assert "confidence" in str(messages[-1])
    assert "response_format" not in calls[0]
    assert response.structured_output == {"target": "Save", "confidence": 0.9}


def test_prompt_schema_joins_the_cacheable_system_prefix(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response('{"target":"Save","confidence":0.9}')

    request = _request(tmp_path, structured=True).model_copy(
        update={
            "messages": (
                ModelMessage(role=ModelMessageRole.SYSTEM, content=(
                    TextModelContent(text="Stable planner instructions."),
                )),
                ModelMessage(role=ModelMessageRole.USER, content=(
                    TextModelContent(text="Changing task context."),
                )),
            )
        }
    )
    _client(tmp_path, complete).complete(request, StructuredAnswer)
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "confidence" in str(messages[0]["content"])
    assert "Changing task context." not in str(messages[0]["content"])
    assert messages[1]["content"] == "Changing task context."


def test_native_schema_mode_uses_shared_litellm_argument(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response('{"target":"Save","confidence":0.9}')

    response = _client(
        tmp_path,
        complete,
        structured_output_mode=StructuredOutputMode.NATIVE_SCHEMA,
    ).complete(_request(tmp_path, structured=True), StructuredAnswer)

    assert calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured-answer",
            "schema": StructuredAnswer.model_json_schema(),
            "strict": True,
        },
    }
    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    prompt_reference = response.calls[0].prompt_artifact
    assert prompt_reference is not None
    prompt_payload = _artifact_payload(tmp_path, prompt_reference)
    request_options = prompt_payload["request_options"]
    assert isinstance(request_options, dict)
    assert request_options["response_format"] == calls[0]["response_format"]


@pytest.mark.parametrize(
    ("provider_response", "disposition"),
    [
        (_response(""), ModelCallDisposition.EMPTY),
        (_response(None, refusal="blocked"), ModelCallDisposition.REFUSED),
        (_response("partial", finish_reason="length"), ModelCallDisposition.TRUNCATED),
        ({"choices": []}, ModelCallDisposition.MALFORMED),
    ],
)
def test_nontransient_model_outcomes_are_not_retried(
    tmp_path: Path,
    provider_response: dict[str, object],
    disposition: ModelCallDisposition,
) -> None:
    call_count = 0

    def complete(**arguments: object) -> object:
        nonlocal call_count
        del arguments
        call_count += 1
        return provider_response

    response = _client(tmp_path, complete).complete(_request(tmp_path))

    assert response.disposition is disposition
    assert call_count == 1
    assert len(response.calls) == 1


def test_transient_failures_are_retried_and_each_call_is_recorded(tmp_path: Path) -> None:
    responses: list[object] = [TimeoutError("prompt content"), _response()]
    waits: list[float] = []

    def complete(**arguments: object) -> object:
        del arguments
        result = responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    response = _client(tmp_path, complete, waits=waits).complete(_request(tmp_path))

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert [call.attempt_number for call in response.calls] == [1, 2]
    assert response.calls[0].disposition is ModelCallDisposition.PROVIDER_ERROR
    assert response.calls[1].disposition is ModelCallDisposition.COMPLETED
    assert waits == [1]


def test_terminal_provider_failure_is_safe_and_not_retried(tmp_path: Path) -> None:
    def complete(**arguments: object) -> object:
        del arguments
        raise RuntimeError("The provider quoted hunter2 and the prompt.")

    response = _client(tmp_path, complete).complete(_request(tmp_path))

    assert response.disposition is ModelCallDisposition.PROVIDER_ERROR
    assert response.reason == "RuntimeError"
    assert len(response.calls) == 1
    call_reference = response.calls[0].call_artifact
    assert call_reference is not None
    assert "hunter2" not in (tmp_path / call_reference.path).read_text()


def test_image_message_uses_verified_png_only_in_provider_request(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response()

    request = _request(tmp_path, image=True)
    image_content = request.messages[0].content[1]
    assert isinstance(image_content, ImageModelContent)
    response = _client(tmp_path, complete).complete(request)

    messages = calls[0]["messages"]
    assert isinstance(messages, list)
    assert "data:image/png;base64," in str(messages)
    prompt_reference = response.calls[0].prompt_artifact
    assert prompt_reference is not None
    prompt_payload = _artifact_payload(tmp_path, prompt_reference)
    assert "iVBOR" not in str(prompt_payload)
    provider_parts = messages[0]["content"]
    artifact_messages = prompt_payload["messages"]
    assert isinstance(provider_parts, list)
    assert isinstance(artifact_messages, list)
    artifact_parts = artifact_messages[0]["content"]
    assert isinstance(artifact_parts, list)
    assert provider_parts[0] == artifact_parts[0]
    assert provider_parts[1]["type"] == "image_url"
    assert artifact_parts[1] == {
        "type": "image_ref",
        "artifact_id": "shot-1",
        "sha256": image_content.screenshot.sha256,
        "width_px": 3,
        "height_px": 2,
    }


def test_contract_mismatch_fails_before_provider_or_artifact_work(tmp_path: Path) -> None:
    provider_called = False

    def complete(**arguments: object) -> object:
        nonlocal provider_called
        del arguments
        provider_called = True
        return _response()

    class DifferentAnswer(BaseModel):
        value: int

    with pytest.raises(ValueError, match="does not match"):
        _client(tmp_path, complete).complete(
            _request(tmp_path, structured=True),
            DifferentAnswer,
        )

    assert provider_called is False
    assert not (tmp_path / "assets/models").exists()


def test_prompt_artifact_matches_prepared_request_and_has_stable_identity(
    tmp_path: Path,
) -> None:
    provider_calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_calls.append(arguments)
        return _response('{"target":"Save","confidence":0.9}')

    client = _client(tmp_path, complete)
    first = client.complete(_request(tmp_path, structured=True), StructuredAnswer)
    second = client.complete(
        _request(tmp_path, structured=True).model_copy(update={"request_id": "request-2"}),
        StructuredAnswer,
    )
    first_reference = first.calls[0].prompt_artifact
    second_reference = second.calls[0].prompt_artifact
    assert first_reference is not None
    assert second_reference is not None
    first_payload = _artifact_payload(tmp_path, first_reference)

    assert first_payload["messages"] == provider_calls[0]["messages"]
    prepared_messages = first_payload["messages"]
    assert isinstance(prepared_messages, list)
    assert "confidence" in str(prepared_messages[-1])
    assert first_payload["request_options"] == {
        name: value for name, value in provider_calls[0].items() if name != "messages"
    }
    assert first_reference.content_sha256 == second_reference.content_sha256


def test_invalid_structured_response_gets_one_recorded_correction(
    tmp_path: Path,
) -> None:
    responses = [
        _response('{"target":"Save"}'),
        _response('{"target":"Save","confidence":0.9}'),
    ]
    provider_calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_calls.append(arguments)
        return responses.pop(0)

    response = _client(tmp_path, complete).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert [call.disposition for call in response.calls] == [
        ModelCallDisposition.MALFORMED,
        ModelCallDisposition.COMPLETED,
    ]
    assert [call.validation_attempt_number for call in response.calls] == [1, 2]
    assert [call.attempt_number for call in response.calls] == [1, 2]
    assert len(provider_calls) == 2
    assert "previous reply did not match" in str(provider_calls[1]["messages"])
    correction_messages = provider_calls[1]["messages"]
    assert isinstance(correction_messages, list)
    assert correction_messages[1] == {
        "role": "assistant",
        "content": '{"target":"Save"}',
    }
    assert all(call.call_artifact is not None for call in response.calls)
    first_prompt = response.calls[0].prompt_artifact
    second_prompt = response.calls[1].prompt_artifact
    assert first_prompt is not None
    assert second_prompt is not None
    assert first_prompt.content_sha256 != second_prompt.content_sha256


def test_validation_retry_reuses_one_verified_image_payload(tmp_path: Path) -> None:
    responses = [
        _response('{"target":"Save"}'),
        _response('{"target":"Save","confidence":0.9}'),
    ]
    reader = CountingImageReader(tmp_path)

    response = _client(
        tmp_path,
        lambda **unused: responses.pop(0),
        image_reader=reader,
    ).complete(_request(tmp_path, structured=True, image=True), StructuredAnswer)

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert reader.read_count == 1


def test_call_artifact_scrubs_nested_cost_details(tmp_path: Path) -> None:
    provider_response = _response()
    provider_response["usage"] = {
        "cost_details": {"provider_note": "hunter2"},
    }

    response = _client(
        tmp_path,
        lambda **unused: provider_response,
        scrub_text=lambda text: text.replace("hunter2", "{{secret}}"),
    ).complete(_request(tmp_path))

    reference = response.calls[0].call_artifact
    assert reference is not None
    artifact = (tmp_path / reference.path).read_text()
    assert "hunter2" not in artifact
    assert "{{secret}}" in artifact


def test_missing_credentials_fail_during_client_construction(tmp_path: Path) -> None:
    provider_called = False

    def complete(**arguments: object) -> object:
        nonlocal provider_called
        del arguments
        provider_called = True
        return _response()

    with pytest.raises(ModelInputError, match="EXAMPLE_API_KEY"):
        _client(
            tmp_path,
            complete,
            credential_check=lambda _: ("EXAMPLE_API_KEY",),
        )

    assert provider_called is False
    assert not (tmp_path / "assets/models").exists()
