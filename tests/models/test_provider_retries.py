"""Provider-directed retry timing at the model boundary."""

from pathlib import Path

from yewo.models import ModelCallDisposition

from models.test_client import _client, _request, _response


def test_rate_limit_wait_honors_provider_retry_after(tmp_path: Path) -> None:
    class RateLimitFailure(RuntimeError):
        status_code = 429

        def __init__(self) -> None:
            super().__init__("sensitive provider message")
            self.response = type("ProviderResponse", (), {"headers": {"Retry-After": "7"}})()

    responses: list[object] = [RateLimitFailure(), _response()]
    waits: list[float] = []

    def complete(**arguments: object) -> object:
        del arguments
        result = responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    response = _client(tmp_path, complete, waits=waits).complete(_request(tmp_path))

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert len(response.calls) == 2
    assert waits == [7]
