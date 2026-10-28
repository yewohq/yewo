"""Yewo describes its cache boundary and leaves provider translation to LiteLLM.

A run measured 487,503 prompt tokens with `cached_tokens` zero on every one of
its 61 calls, because no request said where its stable prefix ended. The
boundary is now declared on every call, whatever the route. Which marker a
provider reads, and whether it reads one at all, is LiteLLM's translation.
"""

import unittest
from unittest import mock

from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import SystemMessage, TextMessage


class PromptCacheBoundaryTests(unittest.TestCase):
    def setUp(self):
        answer = mock.Mock()
        answer.choices = [mock.Mock(message=mock.Mock(content="done"))]
        messages = [
            SystemMessage(content="Action schema, constraints, and the goal."),
            TextMessage(content="Elements 1 through 40."),
        ]
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(model="example/model").generate_response(messages)
        self.request = completion.call_args.kwargs

    def test_the_request_declares_where_the_stable_prefix_ends(self):
        self.assertEqual(
            self.request["cache_control_injection_points"],
            [{"location": "message", "role": "system"}],
        )

    def test_yewo_does_not_write_a_provider_cache_marker(self):
        self.assertEqual(
            self.request["messages"],
            [
                {
                    "role": "system",
                    "content": "Action schema, constraints, and the goal.",
                },
                {"role": "user", "content": "Elements 1 through 40."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
