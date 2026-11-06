"""Provider-neutral prompt-prefix caching tests."""

from pathlib import Path

from yewo.models import ModelMessage, ModelMessageRole, TextModelContent

from models.test_client import (
    StructuredAnswer,
    _artifact_payload,
    _client,
    _request,
    _response,
)
from models.support import build_model_profile, build_test_client


def test_client_translates_provider_neutral_cache_prefixes(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response()

    request = _request(tmp_path).model_copy(
        update={
            "messages": (
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text="Stable instructions."),),
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="Stable execution history."),),
                    cache_prefix_end=True,
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="Current observation."),),
                    cache_prefix_end=True,
                ),
            )
        }
    )

    response = _client(tmp_path, complete).complete(request)

    assert calls[0]["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": 1},
        {"location": "message", "index": 2},
    ]
    assert response.calls[0].prompt_artifact is not None
    prompt_payload = _artifact_payload(tmp_path, response.calls[0].prompt_artifact)
    prompt_messages = prompt_payload["messages"]
    assert isinstance(prompt_messages, list)
    assert isinstance(prompt_messages[1], dict)
    assert isinstance(prompt_messages[2], dict)
    assert prompt_messages[1]["cache_prefix_end"] is True
    assert prompt_messages[2]["cache_prefix_end"] is True


def test_prompt_schema_joins_the_cacheable_system_prefix(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response('{"target":"Save","confidence":0.9}')

    request = _request(tmp_path, structured=True).model_copy(
        update={
            "messages": (
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text="Stable planner instructions."),),
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="Changing task context."),),
                ),
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


def test_client_honours_cache_prefixes_without_a_cache_control_role(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        return _response()

    profile = build_model_profile(
        provider="example",
        model="example/grounder",
    )
    request = _request(tmp_path).model_copy(
        update={
            "messages": (
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text="Stable execution history."),),
                    cache_prefix_end=True,
                ),
            )
        }
    )

    build_test_client(tmp_path, complete, profile).complete(request)

    # A profile that sets no cache_control_role must still get its explicit
    # per-message breakpoints, or the whole growing prefix is billed every call.
    assert calls[0]["cache_control_injection_points"] == [{"location": "message", "index": 0}]
