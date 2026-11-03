"""Structured answers that arrive wrapped in model prose."""

from yewo.models import ModelCallDisposition

from models.test_client import StructuredAnswer, _client, _request, _response


def _complete(text: str):
    def complete(**_arguments: object) -> object:
        return _response(text)

    return complete


def test_fenced_answer_after_prose_is_accepted(tmp_path) -> None:
    text = (
        "Looking at this evidence, I need to trace the workflow first.\n\n"
        "**Key sequence**: open the form, then save.\n\n"
        '```json\n{"target":"Save","confidence":0.9}\n```'
    )

    response = _client(tmp_path, _complete(text)).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert response.structured_output == {"target": "Save", "confidence": 0.9}
    assert len(response.calls) == 1


def test_unfenced_answer_after_prose_is_accepted(tmp_path) -> None:
    text = 'Here is my answer.\n{"target":"Save","confidence":0.9}\nLet me know.'

    response = _client(tmp_path, _complete(text)).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert response.structured_output == {"target": "Save", "confidence": 0.9}


def test_prose_without_any_answer_stays_malformed(tmp_path) -> None:
    response = _client(tmp_path, _complete("I cannot determine the target.")).complete(
        _request(tmp_path, structured=True),
        StructuredAnswer,
    )

    assert response.disposition is ModelCallDisposition.MALFORMED


def test_validation_retry_reports_a_missing_field_from_json_after_prose(tmp_path) -> None:
    responses = [
        _response('I will save it.\n{"target":"Save"}'),
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
    correction_messages = provider_calls[1]["messages"]
    assert isinstance(correction_messages, list)
    correction = next(
        message for message in correction_messages if "previous reply" in str(message)
    )
    assert '"loc":["confidence"]' in str(correction)
    assert '"type":"missing"' in str(correction)
    assert '"type":"json_invalid"' not in str(correction)
