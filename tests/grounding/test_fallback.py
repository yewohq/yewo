"""Grounding fail-over across candidate models and their coordinate spaces."""

import pytest
from pydantic import BaseModel, ValidationError

from yewo.grounding import (
    GroundedPoint,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingProfile,
    GroundingService,
    ModelGroundingResolution,
)
from yewo.models import ModelCallDisposition, ModelRequest, ModelResponse, TextModelContent
from grounding.support import build_call, build_grounding_profile, build_grounding_request


class SequencedModelClient:
    """Return one configured response per call and retain every request."""

    def __init__(self, responses: list[tuple[str | None, ModelCallDisposition]]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []
        self.model_responses: list[ModelResponse] = []

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        del response_model, validation_context
        self.requests.append(request)
        text, disposition = self._responses[len(self.requests) - 1]
        response = ModelResponse(
            request_id=request.request_id,
            disposition=disposition,
            text=text,
            reason=None if disposition is ModelCallDisposition.COMPLETED else "rate limited",
            calls=(
                build_call(
                    request.request_id,
                    profile_id=request.profile_id,
                    disposition=disposition,
                ),
            ),
            retention=request.retention,
        )
        self.model_responses.append(response)
        return response


def _chained_profile() -> GroundingProfile:
    return build_grounding_profile(
        fallbacks=[
            {
                "model_profile_id": "grounder-secondary",
                "coordinate_convention": GroundingCoordinateConvention.NORMALIZED_1000,
            }
        ]
    )


def test_an_absent_primary_provider_falls_back_and_decodes_the_second_convention() -> None:
    client = SequencedModelClient(
        [
            (None, ModelCallDisposition.PROVIDER_ERROR),
            ("(250,750)", ModelCallDisposition.COMPLETED),
        ]
    )
    service = GroundingService(client, _chained_profile())

    result = service.ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(result.location, GroundedPoint)
    assert result.location.pixel.model_dump() == {"x": 50, "y": 299}
    assert [request.profile_id for request in client.requests] == [
        "grounder-primary",
        "grounder-secondary",
    ]
    request_profiles = {
        call.request_id: call.profile_id
        for response in client.model_responses
        for call in response.calls
    }
    assert isinstance(result.resolution, ModelGroundingResolution)
    assert result.resolution.model_request_ids[-1] in request_profiles
    assert request_profiles[result.resolution.model_request_ids[-1]] == "grounder-secondary"


def test_each_candidate_is_told_its_own_coordinate_space() -> None:
    client = SequencedModelClient(
        [
            (None, ModelCallDisposition.PROVIDER_ERROR),
            ("(250,750)", ModelCallDisposition.COMPLETED),
        ]
    )

    GroundingService(client, _chained_profile()).ground(build_grounding_request())

    primary, secondary = (request.messages[0].content[0] for request in client.requests)
    assert isinstance(primary, TextModelContent)
    assert isinstance(secondary, TextModelContent)
    assert "source screenshot pixels" in primary.text
    assert "integers from 0 to 1000" in secondary.text


def test_a_refusal_is_an_answer_and_never_shops_the_next_candidate() -> None:
    client = SequencedModelClient(
        [
            ("refused: target not found", ModelCallDisposition.COMPLETED),
            ("(250,750)", ModelCallDisposition.COMPLETED),
        ]
    )

    result = GroundingService(client, _chained_profile()).ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.REFUSED
    assert len(client.requests) == 1


def test_a_malformed_answer_never_shops_the_next_candidate() -> None:
    client = SequencedModelClient(
        [
            ("not a coordinate", ModelCallDisposition.COMPLETED),
            ("(250,750)", ModelCallDisposition.COMPLETED),
        ]
    )

    result = GroundingService(client, _chained_profile()).ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.MALFORMED
    assert len(client.requests) == 1


def test_an_exhausted_chain_reports_the_absent_provider() -> None:
    client = SequencedModelClient(
        [
            (None, ModelCallDisposition.PROVIDER_ERROR),
            (None, ModelCallDisposition.PROVIDER_ERROR),
        ]
    )

    result = GroundingService(client, _chained_profile()).ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.PROVIDER_UNAVAILABLE
    assert len(client.requests) == 2


def test_a_fallback_may_not_repeat_the_model_it_is_meant_to_replace() -> None:
    with pytest.raises(ValidationError, match="different model profile"):
        build_grounding_profile(
            fallbacks=[
                {
                    "model_profile_id": "grounder-primary",
                    "coordinate_convention": GroundingCoordinateConvention.NORMALIZED,
                }
            ]
        )
