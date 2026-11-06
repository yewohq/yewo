"""Provider-directed retry timing at the model boundary."""

from pathlib import Path

from yewo.models import ModelCallDisposition
from yewo.run_narration import RunNarrator, narration_scope

from models.test_client import _client, _request, _response


def test_rate_limit_wait_honors_provider_retry_after_and_reports_progress(tmp_path: Path) -> None:
    class RateLimitFailure(RuntimeError):
        status_code = 429

        def __init__(self, retry_after_seconds: int) -> None:
            super().__init__("sensitive provider message")
            self.response = type(
                "ProviderResponse",
                (),
                {"headers": {"Retry-After": str(retry_after_seconds)}},
            )()

    responses: list[object] = [RateLimitFailure(15), RateLimitFailure(30), _response()]
    waits: list[float] = []

    def complete(**arguments: object) -> object:
        del arguments
        result = responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    with narration_scope(RunNarrator(tmp_path)):
        response = _client(tmp_path, complete, waits=waits).complete(_request(tmp_path))

    assert response.disposition is ModelCallDisposition.COMPLETED
    assert len(response.calls) == 3
    assert waits == [15, 30]
    session = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Waiting 15 seconds before attempt 2." in session
    assert "Waiting 30 seconds before attempt 3." in session
    assert "sensitive provider message" not in session
