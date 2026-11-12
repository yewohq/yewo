"""What Yewo asks a provider to spend on private reasoning before it answers."""

from contextlib import suppress
from pathlib import Path

from yewo.models import (
    ModelMessage,
    ModelMessageRole,
    ModelReasoningEffort,
    ModelReasoningMode,
    ModelRequest,
    ModelRole,
    TextModelContent,
)
from yewo.artifacts import ArtifactRetention
from yewo.models.profiles import ModelGenerationSettings

from models.support import build_model_profile, build_test_client


def _request() -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        profile_id="grounder-primary",
        role=ModelRole.GROUNDER,
        messages=(
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(TextModelContent(text="Select Save."),),
            ),
        ),
        retention=ArtifactRetention.RUN,
    )


def _router_options(tmp_path: Path, generation: ModelGenerationSettings) -> dict[str, object]:
    calls: list[dict[str, object]] = []

    def complete(**arguments: object) -> object:
        calls.append(arguments)
        raise _Stop

    profile = build_model_profile().model_copy(update={"generation": generation})
    with suppress(_Stop):
        build_test_client(tmp_path, complete, profile).complete(_request())
    body = calls[0].get("extra_body")
    assert isinstance(body, dict)
    return body


class _Stop(Exception):
    """Abort the call once the provider arguments are captured."""


def test_a_bounded_role_asks_the_provider_for_that_much_reasoning(tmp_path: Path) -> None:
    # Reasoning and the answer share one completion budget, so a role left to the provider can
    # spend the whole budget thinking and return no answer at all.
    options = _router_options(
        tmp_path,
        build_model_profile().generation.model_copy(
            update={"reasoning_effort": ModelReasoningEffort.MINIMAL}
        ),
    )

    assert options["reasoning"] == {"effort": "minimal"}


def test_a_role_that_does_not_deliberate_asks_for_no_reasoning(tmp_path: Path) -> None:
    options = _router_options(
        tmp_path,
        build_model_profile().generation.model_copy(
            update={"reasoning": ModelReasoningMode.DISABLED}
        ),
    )

    assert options["reasoning"] == {"enabled": False}


def test_an_unset_role_leaves_the_decision_to_the_provider(tmp_path: Path) -> None:
    options = _router_options(tmp_path, build_model_profile().generation)

    assert "reasoning" not in options
