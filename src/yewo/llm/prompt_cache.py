"""Which providers store a prompt prefix, and where a request marks it.

A prompt cache turns the unchanging request prefix into a cheap read. The
planner puts the action schema, constraints, and goal in one system message.
The element list and screenshot follow it. A provider cache keys on this
unbroken prefix.

Anthropic stores nothing unless the request says where that prefix ends, so
the arrangement alone saved nothing. Other providers either cache a prefix
without being asked or read a different request field, and their payloads are
left as they are.

A short prefix is still not stored. Anthropic's floor is around a thousand
tokens on Sonnet-class models and higher on Haiku, and a request under it is
answered normally with nothing cached. The usage record is what says whether a
prefix was stored, not this module.
"""

# From its defining module: the re-export on the package is untyped.
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

CACHE_CONTROL = {"type": "ephemeral"}
"""The mark Anthropic reads as "store the request up to here".

Ephemeral is the lifetime the API offers without a longer term being bought.
Its entry lives about five minutes and every read renews it, which outlasts
the gap between two steps of a run.
"""

BREAKPOINT_PROVIDERS = frozenset({"anthropic"})
"""Providers that store a prefix only when the request marks the end of it.

This list is not based on the capability flag alone. LiteLLM also reports
providers that cache automatically. A mark sent to those providers requests a
different feature.
"""


def mark_cacheable_prefix(payload: list[dict], model: str) -> list[dict]:
    """The payload with a cache mark closing its system message.

    The function returns an unchanged payload when the provider ignores the
    mark. It also preserves system messages that already use content blocks.
    Their builder controls their marks.
    """
    if not should_mark_breakpoint(model):
        return payload
    return [
        _add_breakpoint(message) if _is_cacheable_prefix(message) else message
        for message in payload
    ]


def should_mark_breakpoint(model: str) -> bool:
    """Whether this model's provider reads a cache mark in the request."""
    try:
        provider = get_llm_provider(model)[1]
    except BadRequestError:
        return False
    return provider in BREAKPOINT_PROVIDERS


def _is_cacheable_prefix(message: dict) -> bool:
    return message.get("role") == "system" and isinstance(message.get("content"), str)


def _add_breakpoint(message: dict) -> dict:
    return {
        **message,
        "content": [
            {
                "type": "text",
                "text": message["content"],
                "cache_control": dict(CACHE_CONTROL),
            }
        ],
    }
