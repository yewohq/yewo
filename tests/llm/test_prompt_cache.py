"""Whether a request asks the provider to store its unchanging prefix.

The planner splits its prompt so that the schema, the constraints and the goal
precede the element list and the screenshot, which is only worth doing if the
request marks where that prefix ends. A run measured 487,503 prompt tokens with
`cached_tokens` zero on every one of its 61 calls, because the mark was never
sent.
"""

import unittest
from unittest import mock

from yewo.llm import prompt_cache
from yewo.llm.messages import ImageMessage, SystemMessage, TextMessage


ANTHROPIC_MODEL = "anthropic/claude-sonnet-5"
OPENROUTER_MODEL = "openrouter/google/gemini-2.5-flash"


def _payload() -> list[dict]:
    return [
        SystemMessage(content="Action schema, constraints, and the goal.").to_dict(),
        TextMessage(content="Elements 1 through 40.").to_dict(),
    ]


class BreakpointProviderTests(unittest.TestCase):
    """A mark sent to the wrong provider asks for a feature it does not have."""

    def test_anthropic_reads_the_mark(self):
        self.assertTrue(prompt_cache.should_mark_breakpoint(ANTHROPIC_MODEL))

    def test_openrouter_is_left_to_its_own_caching(self):
        self.assertFalse(prompt_cache.should_mark_breakpoint(OPENROUTER_MODEL))

    def test_an_unrecognized_model_is_not_marked(self):
        """The provider lookup raises on these, so capability is asked first."""
        self.assertFalse(prompt_cache.should_mark_breakpoint("not-a-real-model"))


class CacheablePrefixTests(unittest.TestCase):
    def test_the_system_message_closes_with_a_cache_mark(self):
        marked = prompt_cache.mark_cacheable_prefix(_payload(), ANTHROPIC_MODEL)

        system = marked[0]
        self.assertEqual(
            system["content"],
            [
                {
                    "type": "text",
                    "text": "Action schema, constraints, and the goal.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )

    def test_the_messages_after_the_prefix_are_unchanged(self):
        """A mark further down would key the cache on the screenshot."""
        payload = _payload()

        marked = prompt_cache.mark_cacheable_prefix(payload, ANTHROPIC_MODEL)

        self.assertEqual(marked[1], payload[1])
        self.assertEqual(len(marked), len(payload))

    def test_a_provider_that_caches_on_its_own_receives_no_mark(self):
        payload = _payload()

        self.assertEqual(
            prompt_cache.mark_cacheable_prefix(payload, OPENROUTER_MODEL), payload
        )

    def test_the_caller_s_payload_is_not_mutated(self):
        payload = _payload()

        prompt_cache.mark_cacheable_prefix(payload, ANTHROPIC_MODEL)

        self.assertEqual(payload[0]["content"], "Action schema, constraints, and the goal.")

    def test_a_message_already_sent_as_blocks_keeps_its_own_marks(self):
        blocks = ImageMessage(text="The screen now.", png_bytes=b"png").to_dict()

        marked = prompt_cache.mark_cacheable_prefix([blocks], ANTHROPIC_MODEL)

        self.assertEqual(marked, [blocks])


class RequestedCachingTests(unittest.TestCase):
    """The mark has to reach the provider through the client, not just exist."""

    def _sent_messages(self, model: str) -> list[dict]:
        from yewo.llm.llm_client import LLMClient

        answer = mock.Mock()
        answer.choices = [mock.Mock(message=mock.Mock(content="done"))]
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(model=model).generate_response(
                [SystemMessage(content="Action schema, constraints, and the goal.")]
            )
        return completion.call_args.kwargs["messages"]

    def test_a_call_to_anthropic_asks_for_the_prefix_to_be_stored(self):
        system = self._sent_messages(ANTHROPIC_MODEL)[0]

        self.assertEqual(system["content"][0]["cache_control"], {"type": "ephemeral"})

    def test_a_call_to_openrouter_sends_the_prefix_as_it_was_built(self):
        system = self._sent_messages(OPENROUTER_MODEL)[0]

        self.assertEqual(system["content"], "Action schema, constraints, and the goal.")


if __name__ == "__main__":
    unittest.main()
