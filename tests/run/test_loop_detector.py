"""Interaction repetition follows controls across harmless visual jitter."""

import unittest

from yewo.agent.actuation.actions import Coordinates, TapMarkAction
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.memory.loop_detector import RollingLoopDetector

SCREEN_A = "daa82d505aa82d5058a225545822b04d2d57f2af2d57f2af2d57d2af2d57d2ab"
SCREEN_A_JITTER = "daa82d505aa82d5058a225545823b00d2d57f2af2d57f2af2d57d2af2d57d2ab"
SCREEN_B = "d8a82d5058a021504800f2a92d56d2ab2d57d2ab2d57f22fbc7fa55fdaafa555"
SIZE = (1080, 2400)


def _tap_at(x: int, y: int, target: str = "Add") -> TapCoordinateAction:
    return TapCoordinateAction(
        coordinates=Coordinates(x=x, y=y), target=target, reason="continue"
    )


def _tap_mark(mark: int) -> TapMarkAction:
    return TapMarkAction(mark=mark, target="Add", reason="continue")


class PageStagnationTests(unittest.TestCase):
    def test_visual_jitter_still_counts_as_one_stagnant_screen(self):
        detector = RollingLoopDetector()

        for screen in (SCREEN_A, SCREEN_A_JITTER) * 4:
            detector.record_screen(screen)

        self.assertTrue(detector.page_stagnation_terminated)

    def test_navigation_clears_page_stagnation_termination(self):
        detector = RollingLoopDetector()
        for _ in range(8):
            detector.record_screen(SCREEN_A)

        detector.record_screen(SCREEN_B)

        self.assertFalse(detector.terminated())

    def test_small_changes_cannot_chain_into_a_different_screen(self):
        detector = RollingLoopDetector()
        original = f"{0:064x}"
        near_original = f"{(1 << 9) - 1:064x}"
        different = f"{(1 << 18) - 1:064x}"

        detector.record_screen(original)
        detector.record_screen(near_original)
        detector.record_screen(different)
        for _ in range(6):
            detector.record_screen(different)

        self.assertFalse(detector.page_stagnation_terminated)

    def test_transient_destination_does_not_enter_back_history(self):
        detector = RollingLoopDetector()
        action = _tap_at(961, 2219)
        detector.record_screen(SCREEN_A)
        detector.record_action(action, SCREEN_A, SIZE)
        detector.record_delivery_outcome(action, SCREEN_A, SCREEN_B, SIZE)
        detector.record_screen(SCREEN_B)
        detector.record_screen(SCREEN_A_JITTER)

        self.assertIsNone(detector.expected_back_destination(SCREEN_A_JITTER))

    def test_durable_destination_enters_back_history(self):
        detector = RollingLoopDetector()
        action = _tap_at(961, 2219)
        detector.record_screen(SCREEN_A)
        detector.record_action(action, SCREEN_A, SIZE)
        detector.record_delivery_outcome(action, SCREEN_A, SCREEN_B, SIZE)
        detector.record_screen(SCREEN_B)
        detector.record_screen(SCREEN_B)

        self.assertEqual(detector.expected_back_destination(SCREEN_B), SCREEN_A)


class ActionIdentityTests(unittest.TestCase):
    def _identity(self, action, screen=SCREEN_A, size=SIZE):
        return build_control_identity(action, screen, size)

    def test_renumbered_marks_resolve_to_the_same_control(self):
        from yewo.agent.grounding import GROUNDING_MARKS
        from yewo.agent.grounding.marks import Mark, resolve_mark
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        element = GUIElement(
            element="fab",
            type="yolo_detection",
            confidence=0.9,
            text="",
            bounding_box=BoundingBox(x_min=920, y_min=2180, x_max=1002, y_max=2258),
        )
        identities = []
        for mark_number in (10, 11, 12, 13):
            refusal, resolved = resolve_mark(
                _tap_mark(mark_number),
                [Mark(mark_number, element, point=(961, 2219))],
                GROUNDING_MARKS,
            )
            self.assertIsNone(refusal)
            identities.append(self._identity(resolved))

        self.assertTrue(all(identities[0].matches(item) for item in identities[1:]))

    def test_small_jitter_across_an_old_bucket_boundary_is_one_control(self):
        self.assertTrue(
            self._identity(_tap_at(959, 2219)).matches(
                self._identity(_tap_at(961, 2221))
            )
        )

    def test_scaled_coordinates_are_one_control_across_resolutions(self):
        full = self._identity(_tap_at(960, 2220), size=(1080, 2400))
        half = self._identity(_tap_at(480, 1110), size=(540, 1200))

        self.assertTrue(full.matches(half))

    def test_localized_labels_do_not_change_control_identity(self):
        english = self._identity(_tap_at(961, 2219, "Add schedule"))
        arabic = self._identity(_tap_at(961, 2219, "إضافة جدول"))
        cjk = self._identity(_tap_at(961, 2219, "スケジュールを追加"))

        self.assertTrue(english.matches(arabic))
        self.assertTrue(english.matches(cjk))

    def test_two_controls_stay_distinct(self):
        self.assertFalse(
            self._identity(_tap_at(961, 2219)).matches(
                self._identity(_tap_at(73, 205))
            )
        )

    def test_one_position_on_different_screens_is_distinct(self):
        self.assertFalse(
            self._identity(_tap_at(961, 2219), SCREEN_A).matches(
                self._identity(_tap_at(961, 2219), SCREEN_B)
            )
        )


class RepeatedControlTests(unittest.TestCase):
    def _repeat(self, times: int, screens=None):
        detector = RollingLoopDetector()
        screens = screens or [SCREEN_A] * times
        nudges = [
            detector.record_action(_tap_at(961, 2219), screens[index], SIZE)
            for index in range(times)
        ]
        return nudges, detector

    def test_repeated_control_is_nudged(self):
        nudges, _ = self._repeat(4)

        self.assertIsNotNone(nudges[-1])

    def test_repeated_control_escalates_without_android_instructions(self):
        nudges, detector = self._repeat(5)

        self.assertTrue(detector.action_repetition_terminated)
        self.assertTrue(any("different" in (nudge or "") for nudge in nudges))
        self.assertFalse(any("key_event 4" in (nudge or "") for nudge in nudges))

    def test_repeated_control_terminates_across_screen_hash_jitter(self):
        screens = [SCREEN_A, SCREEN_A_JITTER] * 7
        _, detector = self._repeat(13, screens)

        self.assertTrue(detector.action_repetition_terminated)

    def test_same_position_on_different_screens_has_independent_budgets(self):
        detector = RollingLoopDetector()
        for _ in range(2):
            for screen in (SCREEN_A, SCREEN_B):
                detector.record_action(_tap_at(961, 2219), screen, SIZE)

        self.assertFalse(detector.terminated())

    def test_alternating_controls_exhaust_the_state_budget(self):
        detector = RollingLoopDetector()
        for _ in range(6):
            for point in ((961, 2219), (73, 205), (367, 672)):
                detector.record_action(_tap_at(*point), SCREEN_A, SIZE)

        self.assertTrue(detector.terminated())

    def test_durable_progress_clears_only_the_responsible_control(self):
        detector = RollingLoopDetector()
        fab = _tap_at(961, 2219)
        menu = _tap_at(1021, 204)
        for _ in range(3):
            detector.record_action(fab, SCREEN_A, SIZE)

        detector.record_action(menu, SCREEN_A, SIZE)
        detector.record_delivery_outcome(menu, SCREEN_A, SCREEN_B, SIZE)
        detector.record_screen(SCREEN_B)

        self.assertIn("quarantined", detector.record_action(fab, SCREEN_A, SIZE) or "")
        self.assertFalse(detector.action_repetition_terminated)

    def test_transient_change_does_not_clear_attempts(self):
        detector = RollingLoopDetector()
        fab = _tap_at(961, 2219)
        detector.record_action(fab, SCREEN_A, SIZE)
        detector.record_delivery_outcome(fab, SCREEN_A, SCREEN_B, SIZE)
        detector.record_screen(SCREEN_A)
        detector.record_action(fab, SCREEN_A, SIZE)

        self.assertIn("quarantined", detector.record_action(fab, SCREEN_A, SIZE) or "")

    def test_visible_durable_progress_allows_repeating_a_control(self):
        detector = RollingLoopDetector()
        action = _tap_at(500, 500)
        screens = [SCREEN_A, SCREEN_B, f"{1:064x}", f"{(1 << 32) - 1:064x}"]
        for pre, post in zip(screens, screens[1:], strict=False):
            detector.record_action(action, pre, SIZE)
            detector.record_delivery_outcome(action, pre, post, SIZE)
            detector.record_screen(post)

        self.assertFalse(detector.terminated())

    def test_small_visual_jitter_does_not_clear_repeated_control_evidence(self):
        detector = RollingLoopDetector()
        action = _tap_at(500, 500)

        detector.record_action(action, SCREEN_A, SIZE)
        detector.record_delivery_outcome(action, SCREEN_A, SCREEN_A_JITTER, SIZE)
        detector.record_screen(SCREEN_A_JITTER)
        detector.record_action(action, SCREEN_A_JITTER, SIZE)
        detector.record_delivery_outcome(action, SCREEN_A_JITTER, SCREEN_A, SIZE)
        detector.record_screen(SCREEN_A)
        message = detector.record_action(action, SCREEN_A, SIZE)

        self.assertIn("quarantined", message or "")


class SemanticRouteCycleTests(unittest.TestCase):
    def test_returning_through_other_screens_with_one_obligation_opens_recovery(self):
        detector = RollingLoopDetector()

        messages = [
            detector.record_semantic_state(screen, "4:add exercise")
            for screen in (SCREEN_A, SCREEN_B, f"{1:064x}", SCREEN_A)
        ]

        self.assertIn("earlier screen", messages[-1] or "")
        self.assertIsNotNone(detector.take_semantic_cycle())
        self.assertFalse(detector.terminated())

    def test_a_second_equivalent_route_cycle_terminates(self):
        detector = RollingLoopDetector()
        route = (SCREEN_A, SCREEN_B, f"{1:064x}", SCREEN_A)

        for screen in (*route, SCREEN_B, f"{1:064x}", SCREEN_A):
            detector.record_semantic_state(screen, "4:add exercise")

        self.assertTrue(detector.action_repetition_terminated)
        self.assertIn("multi-screen route", detector.termination_reason or "")

    def test_advancing_the_obligation_does_not_form_a_cycle(self):
        detector = RollingLoopDetector()
        detector.record_semantic_state(SCREEN_A, "4:add exercise")
        detector.record_semantic_state(SCREEN_B, "4:add exercise")

        message = detector.record_semantic_state(SCREEN_A, "5:enter settings")

        self.assertIsNone(message)
        self.assertFalse(detector.terminated())


class FocusVerificationLoopTests(unittest.TestCase):
    """Text entry stops by failure class, not by screen text or screenshot identity."""

    def test_third_unverified_focus_terminates(self):
        detector = RollingLoopDetector()

        detector.record_focus_verification_failure()
        detector.record_focus_verification_failure()
        self.assertFalse(detector.focus_verification_terminated)
        detector.record_focus_verification_failure()

        self.assertTrue(detector.focus_verification_terminated)
        self.assertTrue(detector.terminated())

    def test_unrelated_success_does_not_clear_focus_failures(self):
        detector = RollingLoopDetector()
        detector.record_focus_verification_failure()
        detector.record_focus_verification_failure()
        detector.record_success(_tap_at(100, 200), SCREEN_A, SIZE)
        detector.record_focus_verification_failure()

        self.assertTrue(detector.focus_verification_terminated)

    def test_verified_text_entry_clears_focus_failures(self):
        detector = RollingLoopDetector()
        detector.record_focus_verification_failure()
        detector.record_focus_verification_failure()

        detector.record_verified_text_entry()
        detector.record_focus_verification_failure()

        self.assertFalse(detector.focus_verification_terminated)


if __name__ == "__main__":
    unittest.main()
