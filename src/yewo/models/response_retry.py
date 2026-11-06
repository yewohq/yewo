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
    normalized: NormalizedModelCompletion,
    scrub_text: Callable[[str], str],
) -> tuple[ModelMessage, ...]:
    """Build bounded correction messages for one retryable response."""
    if normalized.disposition is ModelCallDisposition.TRUNCATED:
        text = scrub_text(
            "Your previous reply reached its output limit before returning a complete "
            "answer. Return only one concise JSON object that matches the schema."
        )
        return (_user_message(text),)
    if normalized.disposition is ModelCallDisposition.EMPTY:
        text = scrub_text(
            "Your previous reply contained no answer. Return one JSON object that "
            "matches the schema. Keep any private reasoning short."
        )
        return (_user_message(text),)
    if normalized.rejected_text is None:
        text = scrub_text(
            "Your previous reply did not contain usable JSON. Return one JSON object that "
            "matches the schema."
        )
        return (_user_message(text),)
    rejected_text = scrub_text(normalized.rejected_text)
    errors = json.dumps(normalized.validation_errors, default=str, separators=(",", ":"))
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
