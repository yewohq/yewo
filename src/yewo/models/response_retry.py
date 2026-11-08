"""Correct malformed structured responses without changing model contracts."""

from collections.abc import Callable
import json

from yewo.models._litellm_response import NormalizedModelCompletion
from yewo.models.records import (
    ModelCallDisposition,
    ModelMessage,
    ModelMessageRole,
    TextModelContent,
)


def correction_messages(
    completion: NormalizedModelCompletion,
    scrub_text: Callable[[str], str],
) -> tuple[ModelMessage, ...]:
    """Build bounded correction messages for one retryable response."""
    if completion.disposition is ModelCallDisposition.REASONING_EXHAUSTED:
        text = scrub_text(
            "Your previous attempt used its completion budget before producing an answer. "
            "Return only one concise JSON object matching the schema. Keep private reasoning "
            "brief and omit commentary."
        )
        return (_user_message(text),)
    if completion.disposition is ModelCallDisposition.TRUNCATED:
        text = scrub_text(
            "Your previous reply reached its output limit before returning a complete "
            "answer. Return only one concise JSON object that matches the schema."
        )
        return (_user_message(text),)
    if completion.disposition is ModelCallDisposition.EMPTY:
        text = scrub_text(
            "Your previous reply contained no answer. Return one JSON object that "
            "matches the schema. Keep any private reasoning short."
        )
        return (_user_message(text),)
    if completion.rejected_text is None:
        text = scrub_text(
            "Your previous reply did not contain usable JSON. Return one JSON object that "
            "matches the schema."
        )
        return (_user_message(text),)
    rejected_text = scrub_text(completion.rejected_text)
    errors = json.dumps(completion.validation_errors, default=str, separators=(",", ":"))
    text = scrub_text(
        "Your previous reply did not match the required JSON schema:\n"
        f"{errors}\n"
        "Answer again as JSON matching the schema exactly."
    )
    return (
        ModelMessage(
            role=ModelMessageRole.ASSISTANT,
            content=(TextModelContent(text=rejected_text),),
        ),
        _user_message(text),
    )


def _user_message(text: str) -> ModelMessage:
    return ModelMessage(
        role=ModelMessageRole.USER,
        content=(TextModelContent(text=text),),
    )
