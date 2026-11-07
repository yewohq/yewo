"""Bounded recovery for truncated structured model responses."""

from yewo.models import ModelCallDisposition

from models.test_client import StructuredAnswer, _client, _request, _response


def test_structured_client_retries_one_truncated_response(tmp_path) -> None:
    provider_responses = iter(
        (
            _response('{"target":"Sav', finish_reason="length"),
            _response('{"target":"Save","confidence":0.9}'),
        )
    )
    provider_requests: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_requests.append(arguments)
        return next(provider_responses)

    response = _client(tmp_path, complete).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert len(response.calls) == 2
    assert len(provider_requests) == 2
    assert "reached its output limit" in str(provider_requests[1]["messages"])


def test_structured_client_retries_exhausted_reasoning_once(tmp_path) -> None:
    provider_responses = iter(
        (
            _response(None, finish_reason="length"),
            _response('{"target":"Save","confidence":0.9}'),
        )
    )
    provider_requests: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_requests.append(arguments)
        return next(provider_responses)

    response = _client(tmp_path, complete).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert len(provider_requests) == 2
    assert "used its completion budget" in str(provider_requests[1]["messages"])


def test_structured_client_stops_after_one_reasoning_recovery(tmp_path) -> None:
    provider_requests: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_requests.append(arguments)
        return _response(None, finish_reason="length")

    response = _client(tmp_path, complete).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.REASONING_EXHAUSTED
    assert len(provider_requests) == 2


def test_caller_can_own_reasoning_exhaustion_recovery(tmp_path) -> None:
    provider_requests: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        provider_requests.append(arguments)
        return _response(None, finish_reason="length")

    request = _request(tmp_path, structured=True).model_copy(
        update={"retry_reasoning_exhaustion": False}
    )
    response = _client(tmp_path, complete).complete(request, StructuredAnswer)

    assert response.disposition is ModelCallDisposition.REASONING_EXHAUSTED
    assert len(provider_requests) == 1
