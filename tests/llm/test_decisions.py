"""The decision route: its credentials, its record, and what it refuses."""

import unittest
from types import SimpleNamespace
from unittest import mock

from yewo.utils.jev_deps import jev_available


class DecisionCredentialTests(unittest.TestCase):
    def test_a_decision_model_names_the_key_its_sdk_reads(self):
        from yewo.llm.credentials import missing_credentials

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(missing_credentials("typesafe/jev-1.14.0"), ["TYPESAFE_API_KEY"])

    def test_that_key_satisfies_it(self):
        from yewo.llm.credentials import missing_credentials

        with mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "k"}, clear=True):
            self.assertEqual(missing_credentials("typesafe/jev-1.14.0"), [])

    def test_other_models_still_go_to_litellm(self):
        from yewo.llm import credentials

        with mock.patch("litellm.validate_environment") as validate:
            validate.return_value = {"missing_keys": ["OPENROUTER_API_KEY"]}
            missing = credentials.missing_credentials("openrouter/google/gemini-2.5-flash")

        self.assertEqual(missing, ["OPENROUTER_API_KEY"])


class DecisionRecordTests(unittest.TestCase):
    """One record for both routes, so a run's cost stays one number."""

    def _response(self):
        return SimpleNamespace(
            model="jev-1.14.0",
            usage=SimpleNamespace(input_tokens=3220, output_tokens=38),
        )

    def test_a_decision_records_the_same_shape_a_completion_does(self):
        from yewo.llm import usage

        record = usage.record_decision(self._response(), "typesafe/jev-1.14.0", 412.0, 0.000135)

        self.assertEqual(record.model, "typesafe/jev-1.14.0")
        self.assertEqual(record.prompt_tokens, 3220)
        self.assertEqual(record.completion_tokens, 38)
        self.assertEqual(record.total_tokens, 3258)
        self.assertEqual(record.latency_ms, 412.0)
        self.assertAlmostEqual(record.cost, 0.000135)

    def test_it_invents_nothing_the_route_cannot_answer(self):
        from yewo.llm import usage

        record = usage.record_decision(self._response(), "typesafe/jev-1.14.0", 1.0, None)

        self.assertEqual(record.finish_reason, "")
        self.assertEqual(record.cached_tokens, 0)
        self.assertEqual(record.reasoning_tokens, 0)
        self.assertIsNone(record.cost)

    def test_the_tracker_totals_both_routes_together(self):
        from yewo.llm import usage
        from yewo.testing.report.usage_tracker import UsageTotals

        totals = UsageTotals()
        totals.add(usage.record_decision(self._response(), "typesafe/jev-1.14.0", 1.0, 0.0001))
        totals.add(
            usage.CallRecord(
                model="openrouter/google/gemini-2.5-flash",
                served_model="gemini-2.5-flash",
                generation_id="gen-1",
                attempt=1,
                finish_reason="stop",
                latency_ms=2000.0,
                cost=0.003,
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
            )
        )

        self.assertEqual(totals.total_calls, 2)
        self.assertEqual(totals.total_prompt_tokens, 4220)
        self.assertAlmostEqual(totals.total_estimated_cost_usd, 0.0031)


class FrameTests(unittest.TestCase):
    """A box is a fraction of the image, so shrinking cannot move a tap."""

    def _png(self, size):
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", size, "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_a_phone_frame_is_shrunk_within_the_limits(self):
        from yewo.llm.jev_client import MAX_PIXELS, MAX_SIDE, MIN_SIDE, prepare_image

        frame = prepare_image(self._png((1080, 2400)))

        self.assertEqual(frame.original, (1080, 2400))
        self.assertLessEqual(max(frame.sent), MAX_SIDE)
        self.assertGreaterEqual(min(frame.sent), MIN_SIDE)
        self.assertLessEqual(frame.sent[0] * frame.sent[1], MAX_PIXELS)

    def test_a_frame_already_within_the_limits_is_left_alone(self):
        from yewo.llm.jev_client import prepare_image

        frame = prepare_image(self._png((706, 1568)))

        self.assertEqual(frame.sent, (706, 1568))

    def test_a_box_maps_back_to_the_device_not_the_sent_image(self):
        from yewo.llm.jev_client import prepare_image

        frame = prepare_image(self._png((1080, 2400)))

        # The tick in the Fossify header, as the model reports it.
        self.assertEqual(frame.to_device(0.91, 0.08, 0.96, 0.11), (983, 192, 1037, 264))


class GrounderTests(unittest.TestCase):
    def _response(self, confidence, present, edges=(0.91, 0.08, 0.96, 0.11)):
        left, top, right, bottom = edges
        return SimpleNamespace(
            answers={
                "where": SimpleNamespace(
                    box=SimpleNamespace(left=left, top=top, right=right, bottom=bottom),
                    confidence=confidence,
                ),
                "present": SimpleNamespace(noul=present),
            }
        )

    def _grounder(self, response):
        from yewo.agent.grounding.description import DecisionGrounder

        client = mock.Mock()
        client.ask.return_value = response
        return DecisionGrounder(client)

    def _png(self):
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (1080, 2400), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_a_small_target_is_kept_despite_a_low_box_confidence(self):
        """The failure this gate exists to avoid: refusing a correct answer
        because the thing it found is small."""
        placed = self._grounder(self._response(0.33, 0.98)).ground(self._png(), "the tick")

        self.assertIsNotNone(placed)
        self.assertEqual(
            (placed.box.x_min, placed.box.y_min, placed.box.x_max, placed.box.y_max),
            (983, 192, 1037, 264),
        )
        self.assertEqual(placed.point, (1010, 228))
        self.assertEqual(placed.box_confidence, 0.33)

    def test_a_thing_that_is_not_there_is_refused(self):
        placed = self._grounder(self._response(0.10, 0.15)).ground(self._png(), "none")

        self.assertIsNone(placed)

    def test_a_box_covering_the_screen_is_refused(self):
        response = self._response(0.9, 0.99, edges=(0.0, 0.0, 1.0, 1.0))
        placed = self._grounder(response).ground(self._png(), "the screen")

        self.assertIsNone(placed)

    def test_a_placed_target_becomes_an_element_perception_can_hand_on(self):
        placed = self._grounder(self._response(0.92, 0.99)).ground(self._png(), "the tick")
        element = placed.as_element()

        self.assertEqual(element.bounding_box.x_min, 983)
        self.assertEqual(element.confidence, 0.92)
        self.assertEqual(element.text, "")


class PartialResponseTests(unittest.TestCase):
    """The SDK types token counts and box confidence as optional, so a call
    that succeeds must not fail on the way into the record."""

    def test_a_missing_token_count_leaves_the_cost_unknown(self):
        from yewo.llm.jev_client import JevClient

        client = JevClient.__new__(JevClient)
        client._model = "typesafe/jev-1.14.0"
        client._price = 4.2e-08
        client._client = mock.Mock()
        client._client.system_one.return_value = SimpleNamespace(
            model="jev-1.14.0",
            answers={},
            usage=SimpleNamespace(input_tokens=None, output_tokens=None),
        )
        records = []
        client.usage_callback = records.append

        client.ask({"q": {"type": "noul", "instructions": "?"}})

        self.assertIsNone(records[0].cost)
        self.assertEqual(records[0].prompt_tokens, 0)

    def test_a_box_without_a_confidence_is_still_placed(self):
        from yewo.agent.grounding.description import DecisionGrounder

        client = mock.Mock()
        client.ask.return_value = SimpleNamespace(
            answers={
                "where": SimpleNamespace(
                    box=SimpleNamespace(left=0.91, top=0.08, right=0.96, bottom=0.11),
                    confidence=None,
                ),
                "present": SimpleNamespace(noul=0.98),
            }
        )
        placed = DecisionGrounder(client).ground(_blank_png(), "the tick")

        self.assertIsNotNone(placed)
        self.assertIsNone(placed.box_confidence)
        self.assertEqual(placed.as_element().confidence, 0.0)


@unittest.skipUnless(jev_available(), "the jev extra is not installed")
class VendorContractTests(unittest.TestCase):
    """What the installed SDK must still provide. Skipped without the extra,
    so the suite stays runnable on a base install."""

    def test_preflight_names_the_variable_the_sdk_reads(self):
        from typesafe_sdk import constants

        from yewo.llm.credentials import DECISION_KEY

        self.assertEqual(constants.API_KEY_ENV, DECISION_KEY)

    def test_the_sdk_exports_what_the_grounder_needs(self):
        import typesafe_sdk

        for name in ("TypeSafeClient", "Image", "Box", "Noul", "RetryPolicy"):
            self.assertTrue(hasattr(typesafe_sdk, name), f"SDK has no {name}")

    def _requests_for_a_rate_limit(self, policy):
        """How many times the real client asks, against a transport that refuses."""
        import httpx2
        from typesafe_sdk import TypeSafeClient

        asked = []

        def refuse(request):
            asked.append(request.url.path)
            return httpx2.Response(429, json={"error": "slow down"})

        client = TypeSafeClient(
            model="noul-1",
            # The transport never reaches a service, so the key is only what
            # the SDK requires to build a client. Taking it from the
            # environment would make this pass on a machine that happens to
            # hold one, and fail everywhere else.
            api_key="not-a-real-key",
            retry=policy,
            transport=httpx2.MockTransport(refuse),
        )
        # Exercise the real retry policy without paying its wall-clock delay.
        client._retry.sleep = lambda _seconds: None
        import typesafe_sdk

        with self.assertRaises(typesafe_sdk.TypeSafeRateLimitError):
            client.system_one(state="", questions={"q": {"type": "boolean", "prompt": "is it?"}})
        return len(asked)

    def test_the_sdk_retries_on_its_own_unless_told_not_to(self):
        """Why Yewo sets a policy at all. Left alone this is three requests."""
        self.assertGreater(self._requests_for_a_rate_limit(None), 1)

    def test_a_zero_policy_leaves_the_sdk_asking_once(self):
        """So Yewo's own three attempts are three requests, and not nine."""
        from typesafe_sdk import RetryPolicy

        self.assertEqual(self._requests_for_a_rate_limit(RetryPolicy(max_retries=0)), 1)

    def test_the_grounder_builds_its_client_with_that_policy(self):
        """One owner for the retry policy, and `call_with_retry` is it."""
        from unittest import mock

        import typesafe_sdk

        from yewo.llm.jev_client import JevClient

        with mock.patch.object(typesafe_sdk, "TypeSafeClient") as built:
            self.assertIsNotNone(JevClient(model="x/noul-1").client)

        self.assertEqual(built.call_args.kwargs["retry"].max_retries, 0)


class SdkCompatibilityTests(unittest.TestCase):
    def test_the_pinned_floor_is_the_release_that_added_images_and_boxes(self):
        """`Image` and `Box` arrived in 0.7.5; an earlier floor installs an SDK
        this code cannot import."""
        import re
        from pathlib import Path

        declared = re.search(
            r'"typesafe-sdk>=([\d.]+)"',
            Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(),
        )
        self.assertIsNotNone(declared)
        floor = tuple(int(part) for part in declared.group(1).split("."))
        self.assertGreaterEqual(floor, (0, 7, 5))


def _blank_png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1080, 2400), "white").save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()


class PlannerSchemaTests(unittest.TestCase):
    """Every grounding mode needs its own schema in both planners.

    QAPlanner overrides step_response_model, so a mode added to DevicePlanner
    alone leaves QA answering with the wrong kind of target.
    """

    def _models(self, planner_class):
        from yewo.agent.grounding import (
            GROUNDING_COORDINATES,
            GROUNDING_DESCRIBE,
            GROUNDING_MARKS,
        )

        planner = planner_class.__new__(planner_class)
        answered = {}
        for mode in (GROUNDING_COORDINATES, GROUNDING_MARKS, GROUNDING_DESCRIBE):
            planner.grounding = mode
            answered[mode] = planner.step_response_model()
        return answered

    def _target_field(self, model):
        """What the action in this schema names its target with."""
        schema = model.model_json_schema()
        fields = set()
        for definition in schema.get("$defs", {}).values():
            properties = definition.get("properties", {})
            if properties.get("type", {}).get("const") == "tap":
                fields |= set(properties) - {"type", "reason", "count", "interval_ms"}
        return fields

    def test_each_mode_names_its_target_its_own_way(self):
        from yewo.agent.plan.planner import DevicePlanner
        from yewo.testing.qa_planner import QAPlanner

        for planner_class in (DevicePlanner, QAPlanner):
            answered = self._models(planner_class)
            self.assertEqual(
                self._target_field(answered["coordinates"]),
                {"coordinates", "target"},
            )
            self.assertEqual(self._target_field(answered["marks"]), {"mark", "target"})
            self.assertEqual(self._target_field(answered["describe"]), {"target"})

    def test_the_two_planners_never_share_a_schema(self):
        from yewo.agent.plan.planner import DevicePlanner
        from yewo.testing.qa_planner import QAPlanner

        device = self._models(DevicePlanner)
        qa = self._models(QAPlanner)
        for mode in device:
            self.assertIsNot(device[mode], qa[mode], f"{mode} shares a schema")
            self.assertIn("is_scenario_feasible", qa[mode].model_fields)

    def test_infeasibility_requires_a_reason(self):
        from pydantic import ValidationError

        from yewo.agent.plan.schema import DoneAction
        from yewo.testing.qa_planner import QAStepResponse

        with self.assertRaises(ValidationError):
            QAStepResponse(
                current_screen="Login",
                observation="The login form is visible.",
                reasoning="Continue the scenario.",
                action=DoneAction(reason="Stop."),
                is_scenario_feasible=False,
                infeasibility_reason="",
            )
