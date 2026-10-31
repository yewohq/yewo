"""Safe model-transport narration without prompt or response bodies."""

from __future__ import annotations

from yewo.models.profiles import ModelProfile
from yewo.models.records import ModelCallRecord, ModelRequest, ModelResponse
from yewo.run_narration import current_narrator


_SOURCE = "model-transport"


def narrate_model_request(request: ModelRequest, profile: ModelProfile) -> None:
    """Record one logical request without recording message content."""
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        _SOURCE,
        f"Requesting {request.role.value} from {profile.provider} ({profile.model})...",
    )
    contract = request.response_contract.name if request.response_contract is not None else "text"
    narrator.trace(
        _SOURCE,
        f"request={request.request_id} profile={profile.profile_id} role={request.role.value} "
        f"provider={profile.provider} model={profile.model} revision={profile.revision or 'none'} "
        f"contract={contract}",
    )


def trace_model_call(call: ModelCallRecord) -> None:
    """Index one persisted provider attempt and its normalized usage."""
    narrator = current_narrator()
    if narrator is None:
        return
    usage = call.usage
    request_artifact = (
        str(call.prompt_artifact.path) if call.prompt_artifact is not None else "none"
    )
    call_artifact = str(call.call_artifact.path) if call.call_artifact is not None else "none"
    narrator.trace(
        _SOURCE,
        f"request={call.request_id} attempt={call.attempt_number} "
        f"validation_attempt={call.validation_attempt_number} "
        f"provider_attempt={call.provider_attempt_number} "
        f"disposition={call.disposition.value} latency_ms={call.latency_ms or 0:.3f} "
        f"served_model={call.served_model or 'unknown'} "
        f"upstream_provider={call.upstream_provider or 'direct'} "
        f"prompt_tokens={usage.prompt_tokens if usage.prompt_tokens is not None else 'unknown'} "
        f"completion_tokens={usage.completion_tokens if usage.completion_tokens is not None else 'unknown'} "
        f"cached_tokens={usage.cached_tokens if usage.cached_tokens is not None else 'unknown'} "
        f"cost_usd={usage.cost_usd if usage.cost_usd is not None else 'unknown'} "
        f"request_artifact={request_artifact} call_artifact={call_artifact}",
    )


def narrate_model_response(request: ModelRequest, response: ModelResponse) -> None:
    """Record the normalized conclusion of one logical request."""
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        _SOURCE,
        f"{request.role.value} model {response.disposition.value} after "
        f"{len(response.calls)} provider call(s).",
    )
    narrator.trace(
        _SOURCE,
        f"request={request.request_id} final_disposition={response.disposition.value} "
        f"call_count={len(response.calls)}",
    )
