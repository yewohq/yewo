"""Declared values stop at the model boundary and resolve only on the device."""

import base64
import io
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from PIL import Image

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.actuation.orchestrator import ActionOrchestrator
from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.device.launch import SetupMode
from yewo.agent.device.platform import Platform
from yewo.agent.grounding.placement import Placement
from yewo.agent.grounding.runtime import (
    GroundingModels,
    ground_and_protect_typed_secret,
)
from yewo.agent.plan.planner import DevicePlanner
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.testing.judge.qa_auditor import QAAuditor, ScreenAssessment
from yewo.testing.qa_planner import QAPlanner, QAStepResponse
from yewo.utils.secrets import (
    AppValueRendering,
    ScreenshotMaskType,
    SecretScrubber,
)


SECRET = "hunter2"
MASK_MOSS = (47, 58, 43)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(output, format="PNG")
    return output.getvalue()


def _secret_element() -> GUIElement:
    return GUIElement(
        element="TextInput",
        bounding_box=BoundingBox(x_min=5, y_min=5, x_max=70, y_max=30),
        confidence=1.0,
        type="field",
        text=SECRET,
    )


def _scrubber() -> SecretScrubber:
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"password": SECRET})
    return scrubber


def _message_text(message) -> str:
    content = message.to_dict()["content"]
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if part.get("type") == "text")


def _messages_text(messages) -> str:
    return " ".join(_message_text(message) for message in messages)


def _message_image(message) -> Image.Image:
    image_part = next(
        part for part in message.to_dict()["content"] if part.get("type") == "image_url"
    )
    encoded = image_part["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


class PlannerTransportTests(unittest.TestCase):
    def test_prompt_and_screenshot_are_redacted_before_the_call(self):
        response = QAStepResponse.model_validate(
            {
                "current_screen": "Login",
                "observation": "Password field",
                "reasoning": "Enter the declared value",
                "action": {
                    "type": "type",
                    "text": "$PASSWORD",
                    "coordinates": [20, 20],
                    "target": "Password field",
                    "reason": "Sign in",
                },
            }
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=response))
        planner = QAPlanner(llm, SetupMode.PRESERVE, scrubber=_scrubber())

        planner.plan_step(
            goal=[f"Enter {SECRET}."],
            gui_elements=[_secret_element()],
            current_image_bytes=_png(),
        )

        messages = llm.generate_response.call_args.args[0]
        self.assertNotIn(SECRET, _messages_text(messages))
        self.assertIn("$PASSWORD", _messages_text(messages))
        self.assertEqual(_message_image(messages[-1]).getpixel((60, 25)), MASK_MOSS)

    def test_base_planner_uses_the_same_transport_boundary(self):
        llm = SimpleNamespace(generate_response=Mock())
        planner = DevicePlanner(llm, scrubber=_scrubber())

        messages = planner._model_messages(
            f"Enter {SECRET}", "Current state", _png(), [_secret_element()]
        )

        self.assertNotIn(SECRET, _messages_text(messages))
        self.assertEqual(_message_image(messages[-1]).getpixel((60, 25)), MASK_MOSS)

    def test_planner_instructions_precede_the_changing_screenshot(self):
        planner = DevicePlanner(SimpleNamespace(generate_response=Mock()), scrubber=_scrubber())

        messages = planner._model_messages("Stable instructions", "Changing state", _png(), [])

        self.assertEqual(messages[0].role, "system")
        self.assertTrue(messages[0].content.startswith("Stable instructions\n\n"))
        self.assertIn("`yewo_secret_mask`", messages[0].content)
        self.assertEqual(messages[1].parts[0]["text"], "Changing state")
        self.assertEqual(messages[1].parts[-1]["type"], "image_url")

    def test_typed_field_is_masked_when_ocr_misses_the_value(self):
        scrubber = _scrubber()
        field = _secret_element().model_copy(update={"text": ""})
        container = field.model_copy(
            update={
                "element": "Container",
                "bounding_box": BoundingBox(x_min=0, y_min=0, x_max=80, y_max=40),
                "type": "container",
            }
        )
        scrubber.protect_typed_region("$PASSWORD", Coordinates(x=20, y=20), [container, field])
        planner = DevicePlanner(SimpleNamespace(generate_response=Mock()), scrubber=scrubber)

        messages = planner._model_messages("Continue", "Current state", _png(), [container, field])

        self.assertEqual(_message_image(messages[-1]).getpixel((60, 25)), MASK_MOSS)
        self.assertEqual(_message_image(messages[-1]).getpixel((75, 35)), (255, 255, 255))
        self.assertEqual(
            scrubber.protected_regions()[0].mask_type,
            ScreenshotMaskType.SECRET,
        )

    def test_typed_field_region_ends_when_navigation_leaves_the_screen(self):
        scrubber = _scrubber()
        field = _secret_element().model_copy(update={"text": ""})
        scrubber.protect_typed_region("$PASSWORD", Coordinates(x=20, y=20), [field])
        images = RedactedImages(scrubber)

        images.clear_after_navigation(ScanInfo(mode="detect", reason="navigation", phash="next"))

        self.assertEqual(scrubber.protected_regions(), ())

    def test_settled_field_bounds_are_masked_after_keyboard_reflow(self):
        scrubber = _scrubber()
        field = _secret_element().model_copy(update={"text": ""})
        scrubber.protect_typed_region("$PASSWORD", Coordinates(x=20, y=20), [field])
        scrubber.protect_typed_bounds(SECRET, (5, 20, 70, 39))
        images = RedactedImages(scrubber)

        redacted = Image.open(io.BytesIO(images.redact_model_image(_png(), [field])))

        self.assertEqual(redacted.getpixel((60, 35)), MASK_MOSS)

    def test_settled_field_keeps_secret_free_app_rendering_evidence(self):
        scrubber = _scrubber()

        scrubber.protect_typed_bounds(
            SECRET,
            (5, 20, 70, 39),
            AppValueRendering.MASKED,
        )

        self.assertEqual(
            scrubber.app_renderings(),
            {"$PASSWORD": AppValueRendering.MASKED},
        )


class AuditorTransportTests(unittest.TestCase):
    def test_auditor_masks_the_image_and_reports_the_exact_leak_locally(self):
        llm = SimpleNamespace(
            generate_response=Mock(
                side_effect=[
                    ScreenAssessment(screen_name="Login"),
                    ScreenAssessment(screen_name="Login"),
                ]
            )
        )
        auditor = QAAuditor(
            llm,
            [],
            [f"Never expose {SECRET}"],
            scrubber=_scrubber(),
        )

        assessment = auditor.audit(_png(), "screen", [_secret_element()])

        messages = llm.generate_response.call_args.args[0]
        self.assertNotIn(SECRET, _messages_text(messages))
        self.assertEqual(_message_image(messages[-1]).getpixel((60, 25)), MASK_MOSS)
        self.assertEqual(len(assessment.sensitive_findings), 1)
        self.assertEqual(assessment.sensitive_findings[0].quoted, "$PASSWORD")

    def test_a_queued_screen_keeps_the_redaction_state_from_when_it_was_read(self):
        llm = SimpleNamespace(
            generate_response=Mock(return_value=ScreenAssessment(screen_name="Splash"))
        )
        scrubber = _scrubber()
        auditor = QAAuditor(llm, [], [], scrubber=scrubber)
        prepared = auditor.prepare_model_image(_png(), [])

        field = _secret_element().model_copy(update={"text": ""})
        scrubber.protect_typed_region("$PASSWORD", Coordinates(x=20, y=20), [field])
        auditor.audit(
            _png(),
            "cached-splash",
            [],
            prepared_model_image=prepared,
        )

        message = llm.generate_response.call_args.args[0][-1]
        self.assertEqual(_message_image(message).getpixel((60, 25)), (255, 255, 255))

    def test_local_rendering_evidence_is_sent_without_the_value(self):
        scrubber = _scrubber()
        scrubber.protect_typed_bounds(
            SECRET,
            (5, 5, 70, 30),
            AppValueRendering.MASKED,
        )
        llm = SimpleNamespace(
            generate_response=Mock(return_value=ScreenAssessment(screen_name="Login"))
        )

        QAAuditor(llm, [], ["Password stays masked"], scrubber=scrubber).audit(
            _png(), "screen", []
        )

        sent = _messages_text(llm.generate_response.call_args.args[0])
        self.assertIn("$PASSWORD: app_rendering=masked", sent)
        self.assertNotIn(SECRET, sent)


class DeviceResolutionTests(unittest.TestCase):
    def test_placeholder_is_resolved_only_when_typing(self):
        device = SimpleNamespace(
            screen_width=100,
            screen_height=100,
            type=Mock(),
            driver=None,
            platform=Platform.ANDROID,
        )
        action = TypeAction(
            text="$PASSWORD",
            coordinates=Coordinates(x=20, y=20),
            reason="Enter the declared value",
        )

        ActionOrchestrator(device, _scrubber()).execute(action)

        device.type.assert_called_once_with(SECRET, 20, 20)

    def test_detector_free_secret_typing_uses_the_grounded_field(self):
        scrubber = _scrubber()
        bounds = BoundingBox(x_min=76, y_min=984, x_max=1026, y_max=1176)
        action = TypeAction(
            text="$PASSWORD",
            coordinates=Coordinates(x=540, y=930),
            target="Password field",
            reason="Enter the declared value",
        )
        snapshot = SimpleNamespace(raw_img_bytes=b"screen", gui_elements=[])
        images = SimpleNamespace(redact_model_image=Mock(return_value=b"safe screen"))
        grounder = Mock(return_value=Placement(box=bounds, present=1.0, box_confidence=0.87))
        grounding_models = cast(
            GroundingModels,
            SimpleNamespace(replay=lambda: SimpleNamespace(ground=grounder)),
        )

        resolved, target_bounds = ground_and_protect_typed_secret(
            action, snapshot, scrubber, images, grounding_models
        )

        self.assertEqual(resolved.coordinates, Coordinates(x=551, y=1080))
        self.assertEqual(target_bounds, bounds)
        grounder.assert_called_once_with(b"safe screen", "Password field")
        self.assertEqual(
            scrubber.protected_regions()[0].bounds,
            (76, 984, 1026, 1176),
        )


class SecretDisplayTests(unittest.TestCase):
    def test_last_four_is_the_only_value_fragment_sent(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets(
            {"card": "4111222233334444"},
            {"card": "last4"},
            allow_model_exposure=True,
        )

        visible = scrubber.for_model("Card 4111222233334444")

        self.assertEqual(visible, "Card $CARD[…4444]")
        self.assertNotIn("411122223333", visible)

    def test_plain_placeholder_also_resolves_under_last_four_policy(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"card": "4111222233334444"}, {"card": "last4"})

        self.assertEqual(scrubber.resolve_model_text("$CARD"), "4111222233334444")

    def test_show_is_an_explicit_transport_opt_out(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"note": SECRET}, {"note": "show"}, allow_model_exposure=True)

        self.assertEqual(scrubber.for_model(SECRET), SECRET)
        self.assertEqual(scrubber.model_labels_in(SECRET), [])
        self.assertEqual(scrubber.scrub(SECRET), "{{note}}")

    def test_exposure_policy_is_ignored_without_the_command_line_gate(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"note": SECRET}, {"note": "show"})

        self.assertEqual(scrubber.for_model(SECRET), "$NOTE")

    def test_unsafe_model_opt_out_does_not_unmask_saved_screenshots(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"note": SECRET}, {"note": "show"}, allow_model_exposure=True)
        field = _secret_element().model_copy(update={"text": ""})
        scrubber.protect_typed_region(SECRET, Coordinates(x=20, y=20), [field])
        images = RedactedImages(scrubber)

        model_image = Image.open(io.BytesIO(images.redact_model_image(_png(), [field])))
        artifact_image = images.redact_artifact_image(Image.open(io.BytesIO(_png())), [field])

        self.assertEqual(model_image.getpixel((60, 25)), (255, 255, 255))
        self.assertEqual(artifact_image.getpixel((60, 25)), MASK_MOSS)


if __name__ == "__main__":
    unittest.main()
