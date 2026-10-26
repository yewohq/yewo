"""Which providers store a prompt prefix, and where a request marks it.

A prompt cache turns the unchanging head of a request into a cheap read. The
planner already arranges for one: the action schema, the constraints and the
goal go in a single system message ahead of the element list and the
screenshot, because a provider cache keys on an unbroken prefix.

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
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import supports_prompt_caching

CACHE_CONTROL = {"type": "ephemeral"}
"""The mark Anthropic reads as "store the request up to here".

Ephemeral is the lifetime the API offers without a longer term being bought.
Its entry lives about five minutes and every read renews it, which outlasts
the gap between two steps of a run.
"""

BREAKPOINT_PROVIDERS = frozenset({"anthropic"})
"""Providers that store a prefix only when the request marks the end of it.

Named rather than taken from the capability flag alone. LiteLLM also reports
prompt caching for providers that cache on their own, and a mark sent to one
of those asks for a different feature than the one meant here.
"""


def mark_cacheable_prefix(payload: list[dict], model: str) -> list[dict]:
    """The payload with a cache mark closing its system message.

    Returned unchanged for a model whose provider does not read the mark, and
    for a system message already sent as content blocks, which carries
    whatever marks its builder chose.
    """
    if not should_mark_breakpoint(model):
        return payload
    return [
        _add_breakpoint(message) if _is_cacheable_prefix(message) else message
        for message in payload
    ]


def should_mark_breakpoint(model: str) -> bool:
    """Whether this model's provider reads a cache mark in the request."""
    # Asked in this order because an unrecognized model raises from the
    # provider lookup and answers False here.
    if not supports_prompt_caching(model=model):
        return False
    return get_llm_provider(model)[1] in BREAKPOINT_PROVIDERS


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
