"""Safe two-tier narration for model transport."""

from pathlib import Path

from yewo.run_narration import RunNarrator, narration_scope
from yewo.models.narration import trace_model_clock_gap

from models.test_client import _client, _request, _response


def test_model_transport_narrates_attribution_and_usage_without_bodies(tmp_path: Path) -> None:
    private_response = "private-response-body-7391"
    with narration_scope(RunNarrator(tmp_path)):
        _client(tmp_path, lambda **unused: _response(private_response)).complete(_request(tmp_path))

    session = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Requesting grounder from openrouter (openrouter/example/grounder)" in session
    assert "grounder model completed after 1 provider call(s)" in session
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "request=request-1 profile=grounder-primary role=grounder" in trace
    assert "served_model=example/grounder-revision-1" in trace
    assert "prompt_tokens=10 completion_tokens=2" in trace
    assert "final_disposition=completed call_count=1" in trace
    assert private_response not in session
    assert private_response not in trace
    assert "Select Save" not in session
    assert "Select Save" not in trace


def test_model_transport_traces_likely_host_suspension(tmp_path: Path) -> None:
    with narration_scope(RunNarrator(tmp_path)):
        trace_model_clock_gap("request-1", wall_seconds=1_060, active_seconds=10)

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "request=request-1 wall_seconds=1060.000 active_seconds=10.000" in trace
    assert "clock_gap_seconds=1050.000" in trace


def test_model_transport_traces_provider_retry_delay(tmp_path: Path) -> None:
    responses: list[object] = [TimeoutError("provider timed out"), _response()]

    def complete(**unused):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    with narration_scope(RunNarrator(tmp_path)):
        _client(tmp_path, complete).complete(_request(tmp_path))

    session = (tmp_path / "runtime/yewo-session.log").read_text()
    assert (
        "grounder provider attempt 1 of 3 failed: "
        "TimeoutError: Provider request timed out. "
        "Waiting 1 second before attempt 2."
    ) in session
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "request=request-1 failed_provider_attempt=1 next_provider_attempt=2" in trace
    assert "provider_retry_delay_seconds=1.000 reason=TimeoutError" in trace
