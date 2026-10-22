"""Surviving a model service that will not answer.

A suite holds one device across every scenario. A rate limit that ends the run
spends the rest of that device on nothing, and a bad API key that is retried
spends it three times over. So the two are told apart, and they end
differently: one scenario, or the run.
"""

import unittest
from unittest import mock

import httpx

from yewo.utils.jev_deps import jev_available
from fakes import rate_limited, refused
from litellm import exceptions
from pydantic import BaseModel, ValidationError

from yewo.errors import Cause, ProviderUnavailable, cause_of
from yewo.llm import failures
from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import TextMessage


class Answer(BaseModel):
    verdict: str


class ClassificationTests(unittest.TestCase):
    """Read from the type and the status, never from the message."""

    def test_a_service_that_could_answer_later_is_retryable(self):
        for exc in (
            rate_limited(),
            exceptions.Timeout("too slow", "m", "openrouter"),
            exceptions.APIConnectionError("dropped", "m", "openrouter"),
            exceptions.InternalServerError("boom", "m", "openrouter"),
            exceptions.ServiceUnavailableError("busy", "m", "openrouter"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertTrue(failures.is_retryable(exc))

    def test_a_request_the_service_refused_is_not_retryable(self):
        """Asking again sends the same request, and it was the request."""
        response = httpx.Response(403, request=httpx.Request("POST", "https://x"))
        for exc in (
            refused(),
            exceptions.BadRequestError("malformed", "m", "openrouter"),
            exceptions.ContextWindowExceededError("too long", "m", "openrouter"),
            exceptions.PermissionDeniedError("no", "m", "openrouter", response),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(failures.is_retryable(exc))

    def test_a_retryable_type_with_no_status_is_still_retried(self):
        """A transport failure can arrive without one, and it is the best case.

        This is what the type list is for. A status list on its own would
        read a dropped connection as a refusal and give up on it.
        """
        dropped = mock.Mock(spec=exceptions.APIConnectionError)
        dropped.status_code = None

        self.assertTrue(failures.is_retryable(dropped))

    def test_a_failure_nothing_recognises_is_not_retried(self):
        """A retry costs the device the suite is holding, so it has to be earned."""
        self.assertFalse(failures.is_retryable(ValueError("something else")))

    def test_each_status_is_decided_as_the_policy_says(self):
        """A gateway reports a status the client named no type for.

        Both lists are written out rather than read off the constant. The
        constant is the policy, and a test that iterated it would agree with
        any edit to it, including a mistake.
        """

        class Gateway(Exception):
            def __init__(self, status):
                self.status_code = status

        for status in (408, 409, 425, 429, 500, 502, 503, 504, 529):
            with self.subTest(retryable=status):
                self.assertTrue(failures.is_retryable(Gateway(status)))

        for status in (400, 401, 402, 403, 404, 422, 451):
            with self.subTest(terminal=status):
                self.assertFalse(failures.is_retryable(Gateway(status)))

    def test_a_description_carries_the_type_and_the_status_only(self):
        """A provider message can quote the prompt back, and this reaches a report."""
        described = failures.describe(rate_limited())

        self.assertIn("RateLimitError", described)
        self.assertIn("429", described)
        self.assertNotIn("slow down", described)

    def test_the_backoff_has_one_pause_fewer_than_it_has_attempts(self):
        """Nothing waits before the first attempt."""
        self.assertEqual(len(failures.BACKOFF_SECONDS), failures.ATTEMPTS - 1)


class RetryTests(unittest.TestCase):
    """The same call, made again while the service could still answer it."""

    def setUp(self):
        self.waited = []
        self.client = LLMClient(model="m", wait=self.waited.append)
        self.messages = [TextMessage(content="hello")]

    def _complete(self, *results):
        """Patches one call per result, raising or returning in order."""
        return mock.patch.object(
            LLMClient, "_complete", side_effect=list(results), autospec=True
        )

    def test_a_rate_limit_that_clears_is_not_a_failure(self):
        with self._complete(rate_limited(), "answered"):
            self.assertEqual(self.client.generate_response(self.messages), "answered")

        self.assertEqual(self.waited, [failures.BACKOFF_SECONDS[0]])

    def test_the_attempts_are_bounded_and_the_waits_grow(self):
        with self._complete(*[rate_limited()] * failures.ATTEMPTS) as call, \
             self.assertRaises(ProviderUnavailable):
            self.client.generate_response(self.messages)

        self.assertEqual(call.call_count, failures.ATTEMPTS)
        self.assertEqual(self.waited, list(failures.BACKOFF_SECONDS))

    def test_nothing_waits_after_the_last_attempt(self):
        """A pause before giving up holds the device and changes no answer."""
        with self._complete(*[rate_limited()] * failures.ATTEMPTS), \
             self.assertRaises(ProviderUnavailable):
            self.client.generate_response(self.messages)

        self.assertEqual(len(self.waited), failures.ATTEMPTS - 1)

    def test_a_refused_request_is_raised_on_the_first_attempt(self):
        with self._complete(refused(), "never reached") as call, \
             self.assertRaises(exceptions.AuthenticationError):
            self.client.generate_response(self.messages)

        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.waited, [])

    def test_the_exhausted_error_names_the_failure_without_quoting_it(self):
        with self._complete(*[rate_limited()] * failures.ATTEMPTS), \
             self.assertRaises(ProviderUnavailable) as raised:
            self.client.generate_response(self.messages)

        message = str(raised.exception)
        self.assertIn("RateLimitError", message)
        self.assertIn("429", message)
        self.assertNotIn("slow down", message)

    def test_the_cause_of_an_exhausted_provider_is_the_provider(self):
        self.assertIs(cause_of(ProviderUnavailable("spent")), Cause.PROVIDER)

    def test_a_schema_mismatch_is_left_to_the_loop_that_owns_it(self):
        """The provider answered. Asking the same question again would not help.

        The outer loop asks a different question, with the validation errors
        in it, so a mismatch must not be retried here as though the service
        had failed.
        """
        mismatch = ValidationError.from_exception_data("Answer", [])
        with self._complete(mismatch, Answer(verdict="ok")) as call:
            answer = self.client.generate_response(self.messages, Answer)

        self.assertEqual(answer.verdict, "ok")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(self.waited, [])


class ReportTextTests(unittest.TestCase):
    """What reaches a report after a provider failed.

    A provider message is not ours. It quotes the request back, and the
    request is the prompt: the screen, the journey, and whatever the app had
    on it at the time. A report is read by people who were not there, and
    kept as an artifact.
    """

    def _reportable(self, error):
        from yewo.testing.runner import reportable

        return reportable(error)

    def test_a_refused_request_is_reduced_to_its_type_and_status(self):
        """Terminal failures reach the report through the abort path."""
        text = self._reportable(
            exceptions.AuthenticationError("key ... for prompt 'Sign in as'", "m", "p")
        )

        self.assertIn("AuthenticationError", text)
        self.assertIn("401", text)
        self.assertNotIn("Sign in as", text)

    def test_a_malformed_request_is_reduced_the_same_way(self):
        text = self._reportable(
            exceptions.BadRequestError("messages[0] said 'card 4111'", "m", "p")
        )

        self.assertNotIn("4111", text)

    def test_yewos_own_provider_failure_keeps_what_yewo_wrote(self):
        """It already names the failure and the attempts, and reducing it loses both."""
        text = self._reportable(
            ProviderUnavailable(
                "The provider did not answer after 3 attempts: "
                "RateLimitError (HTTP 429)"
            )
        )

        self.assertIn("3 attempts", text)
        self.assertIn("429", text)

    def test_every_other_failure_keeps_its_message(self):
        """A device message is what tells a reader what broke."""
        from yewo.errors import DeviceUnavailable

        text = self._reportable(DeviceUnavailable("the emulator stopped responding"))

        self.assertIn("emulator stopped responding", text)

    def test_a_failure_with_no_message_is_named_by_its_type(self):
        self.assertEqual(self._reportable(RuntimeError()), "RuntimeError")


class DecisionModelTests(unittest.TestCase):
    """The grounding model is a provider too, and it rate-limits like one.

    It is reached through its own SDK rather than through the completion
    client, so none of Yewo's policy reached it. The SDK retried on its own
    schedule, which Yewo could neither bound nor report, and a rate limit
    that outlasted it ended the scenario.

    These cases are the policy. `test_decisions.py` holds the SDK to taking
    a zero-retry policy, which is what keeps three attempts three requests.
    """

    def setUp(self):
        self.waited = []
        self.calls = None

    def _asked(self, *results):
        self.calls = mock.Mock(side_effect=list(results))
        return failures.call_with_retry(
            self.calls, "The decision model", wait=self.waited.append
        )

    def test_a_rate_limit_that_clears_is_not_a_failure(self):
        answer = self._asked(rate_limited(), "answered")

        self.assertEqual(answer, "answered")
        self.assertEqual(self.calls.call_count, 2)
        self.assertEqual(self.waited, [failures.BACKOFF_SECONDS[0]])

    def test_the_attempts_are_bounded_the_same_way(self):
        with self.assertRaises(ProviderUnavailable):
            self._asked(*[rate_limited()] * failures.ATTEMPTS)

        self.assertEqual(self.calls.call_count, failures.ATTEMPTS)
        self.assertEqual(self.waited, list(failures.BACKOFF_SECONDS))

    def test_a_refused_request_is_raised_on_the_first_attempt(self):
        with self.assertRaises(exceptions.AuthenticationError):
            self._asked(refused(), "never reached")

        self.assertEqual(self.calls.call_count, 1)
        self.assertEqual(self.waited, [])

    def test_the_exhausted_error_names_the_model_that_would_not_answer(self):
        """Two providers can fail in one run, and a report has to say which."""
        with self.assertRaises(ProviderUnavailable) as raised:
            self._asked(*[rate_limited()] * failures.ATTEMPTS)

        message = str(raised.exception)
        self.assertIn("The decision model", message)
        self.assertIn("429", message)
        self.assertNotIn("slow down", message)


@unittest.skipUnless(jev_available(), "the jev extra is not installed")
class DecisionSdkTests(unittest.TestCase):
    """The decision model's own SDK, classified by its own contract.

    Skipped without the extra, so the suite stays runnable on a base
    install. CI installs the extra, so these run there.

    It reports a status in a field of its own and raises transport errors of
    its own, so every one of its rate limits read as terminal. A 429 from the
    grounding model then ended a scenario that one pause would have saved.
    """

    def _error(self, name, *args):
        import typesafe_sdk

        return getattr(typesafe_sdk, name)(*args)

    def _api_error(self, name, status):
        import httpx2

        return self._error(name, status, {}, httpx2.Headers())

    def test_a_rate_limit_is_read_from_the_field_this_sdk_uses(self):
        """LiteLLM says `status_code` and this one says `status`."""
        self.assertTrue(
            failures.is_retryable(self._api_error("TypeSafeRateLimitError", 429))
        )

    def test_a_server_error_is_retryable_here_too(self):
        self.assertTrue(
            failures.is_retryable(self._api_error("TypeSafeInternalServerError", 503))
        )

    def test_a_rejected_key_is_still_terminal(self):
        self.assertFalse(
            failures.is_retryable(self._api_error("TypeSafeAuthenticationError", 401))
        )

    def test_a_dropped_connection_carries_no_status_and_is_still_retryable(self):
        """It subclasses the builtin, which is what the type list names."""
        self.assertTrue(
            failures.is_retryable(self._error("TypeSafeAPIConnectionError", "dropped"))
        )

    def test_a_timeout_carries_no_status_and_is_still_retryable(self):
        self.assertTrue(
            failures.is_retryable(self._error("TypeSafeAPITimeoutError", 30.0))
        )

    def test_the_status_reaches_the_description(self):
        described = failures.describe(self._api_error("TypeSafeRateLimitError", 429))

        self.assertIn("429", described)
