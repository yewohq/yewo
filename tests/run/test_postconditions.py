import unittest

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.device.focus import FocusedFieldReading
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.run.postconditions import (
    VerificationStatus,
    VerificationClassification,
    ValueVerificationPolicy,
    pending_for,
    specification_for,
    verify,
)
from yewo.utils.secrets import AppValueRendering


def field(text: str) -> GUIElement:
    return GUIElement(
        element="input",
        bounding_box=BoundingBox(x_min=10, y_min=10, x_max=200, y_max=80),
        confidence=0.9,
        type="field",
        text=text,
    )


def action(text: str = "125.00") -> TypeAction:
    return TypeAction(
        text=text,
        coordinates=Coordinates(x=100, y=40),
        target="Declared input",
        reason="Fill the scenario's declared value.",
    )


class EnteredValuePostconditionTests(unittest.TestCase):
    def setUp(self):
        self.values = {"declared_input": "125.00"}
        self.names = ValueNames(self.values)

    def test_a_saved_specification_names_the_value_and_never_contains_it(self):
        specification = specification_for(action(), self.names)

        self.assertIsNotNone(specification)
        assert specification is not None
        dumped = specification.model_dump_json()
        self.assertIn("declared_input", dumped)
        self.assertNotIn("125.00", dumped)

    def test_a_dispatch_receipt_without_settled_evidence_is_unverified(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        result = verify(pending, [])

        self.assertIs(result.status, VerificationStatus.UNVERIFIED)

    def test_a_settled_field_can_confirm_the_entry(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        result = verify(
            pending,
            [],
            FocusedFieldReading("125.00", (10, 10, 200, 80)),
        )

        self.assertIs(result.status, VerificationStatus.PASSED)

    def test_a_delivery_mismatch_requires_agent_recovery(self):
        pending = pending_for(action(), self.names, self.values, "12.50")
        assert pending is not None

        result = verify(
            pending,
            [],
            FocusedFieldReading("12.50", (10, 10, 200, 80)),
        )

        self.assertIs(result.status, VerificationStatus.FAILED)
        self.assertIs(result.classification, VerificationClassification.AGENT_ACTION)
        self.assertNotIn("12.50", result.reason)
        self.assertNotIn("125.00", result.reason)

    def test_an_app_change_after_correct_delivery_is_an_application_failure(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        result = verify(
            pending,
            [],
            FocusedFieldReading("12.50", (10, 10, 200, 80)),
        )

        self.assertIs(result.status, VerificationStatus.FAILED)
        self.assertIs(result.classification, VerificationClassification.APPLICATION)

    def test_target_local_text_can_confirm_when_the_driver_cannot_read_focus(self):
        pending = pending_for(action(), self.names, self.values, None)
        assert pending is not None

        result = verify(pending, [field("125.00")])

        self.assertIs(result.status, VerificationStatus.PASSED)

    def test_text_elsewhere_on_the_screen_is_not_accepted(self):
        elsewhere = field("125.00").model_copy(
            update={
                "bounding_box": BoundingBox(
                    x_min=300,
                    y_min=300,
                    x_max=500,
                    y_max=380,
                )
            }
        )
        pending = pending_for(action(), self.names, self.values, None)
        assert pending is not None

        result = verify(pending, [elsewhere])

        self.assertIs(result.status, VerificationStatus.UNVERIFIED)

    def test_auto_recognizes_an_observed_mask_without_app_specific_configuration(self):
        automatic = pending_for(action(), self.names, self.values, "••••••")
        exact = pending_for(
            action(),
            self.names,
            self.values,
            "••••••",
            verification_policies={"declared_input": "exact"},
        )
        assert automatic is not None
        assert exact is not None

        reading = FocusedFieldReading("••••••", (10, 10, 200, 80))
        self.assertIs(verify(exact, [], reading).status, VerificationStatus.FAILED)
        verified = verify(automatic, [], reading)
        self.assertIs(verified.status, VerificationStatus.PASSED)
        self.assertIs(verified.app_rendering, AppValueRendering.MASKED)

    def test_a_secure_platform_field_is_local_masking_evidence(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        verified = verify(
            pending,
            [],
            FocusedFieldReading("125.00", (10, 10, 200, 80), is_secure=True),
        )

        self.assertIs(verified.status, VerificationStatus.PASSED)
        self.assertIs(verified.app_rendering, AppValueRendering.MASKED)

    def test_exact_ocr_in_a_nonsecure_field_is_local_exposure_evidence(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        verified = verify(
            pending,
            [field("125.00")],
            FocusedFieldReading("125.00", (10, 10, 200, 80), is_secure=False),
        )

        self.assertIs(verified.status, VerificationStatus.PASSED)
        self.assertIs(verified.app_rendering, AppValueRendering.EXPOSED)

    def test_accessibility_value_without_display_metadata_is_not_called_exposed(self):
        pending = pending_for(action(), self.names, self.values, "125.00")
        assert pending is not None

        verified = verify(
            pending,
            [],
            FocusedFieldReading("125.00", (10, 10, 200, 80)),
        )

        self.assertIs(verified.status, VerificationStatus.PASSED)
        self.assertIs(verified.app_rendering, AppValueRendering.UNKNOWN)

    def test_a_permitted_mask_can_keep_the_last_four_characters(self):
        values = {"declared_input": "123456789"}
        pending = pending_for(
            action("123456789"),
            ValueNames(values),
            values,
            "•••••6789",
            verification_policies={
                "declared_input": ValueVerificationPolicy(
                    mode="masked", visible_suffix=4
                )
            },
        )
        assert pending is not None

        reading = FocusedFieldReading("•••••6789", (10, 10, 200, 80))
        self.assertIs(verify(pending, [], reading).status, VerificationStatus.PASSED)

    def test_declared_normalizers_accept_app_formatting(self):
        values = {"declared_input": "125.00"}
        pending = pending_for(
            action(),
            ValueNames(values),
            values,
            "125.00",
            verification_policies={
                "declared_input": ValueVerificationPolicy(
                    mode="normalized", normalizers=("currency", "whitespace")
                )
            },
        )
        assert pending is not None

        reading = FocusedFieldReading("£ 125.00", (10, 10, 200, 80))
        self.assertIs(verify(pending, [], reading).status, VerificationStatus.PASSED)


if __name__ == "__main__":
    unittest.main()
