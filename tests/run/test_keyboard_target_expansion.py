import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.actuation.actions import ShowKeyboardTargetsAction
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.keyboard import observe
from yewo.agent.memory.stall import Stall
from yewo.agent.run import keyboard_targets


class KeyboardTargetExpansionTests(unittest.TestCase):
    def test_marks_planner_has_a_dedicated_keyboard_expansion_action(self):
        from yewo.agent.plan.schema import MarksStepResponse

        response = MarksStepResponse.model_validate(
            {
                "current_screen": "Custom keyboard",
                "observation": "A visual keyboard is visible.",
                "reasoning": "The goal tests the key UI itself.",
                "action": {
                    "type": "show_keyboard_targets",
                    "reason": "Tap a visual key and observe its highlight.",
                },
            }
        )

        self.assertIsInstance(response.action, ShowKeyboardTargetsAction)
        action = response.action
        assert isinstance(action, ShowKeyboardTargetsAction)
        self.assertEqual(action.type.value, "show_keyboard_targets")

    def test_dedicated_request_expands_keys_without_dismissing_keyboard(self):
        key = GUIElement(
            element="AnyKeyClass",
            type="yolo_detection",
            confidence=0.9,
            text="A",
            semantic_roles=frozenset({"keyboard_key"}),
            bounding_box=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50),
        )
        snapshot = SimpleNamespace(
            gui_elements=[key],
            current_phash="screen",
            timestamp="now",
            keyboard_observation=observe([key], native_visible=True),
        )
        action = ShowKeyboardTargetsAction(reason="test the visual key")
        ctx = SimpleNamespace(
            history=Mock(),
            steps=Mock(),
            keyboard_targets_expanded=False,
            pending_loop_nudge=None,
            stall=Stall(),
            last_action_type=None,
        )
        result = keyboard_targets.expand(0, action, snapshot, None, ctx, "marks")

        self.assertIsNone(result)
        self.assertTrue(ctx.keyboard_targets_expanded)
        self.assertIn("now expanded", ctx.pending_loop_nudge)
        ctx.steps.record_action.assert_called_once()
        recorded_result = ctx.steps.record_action.call_args.args[2]
        self.assertTrue(recorded_result.success)
        self.assertIn("without touching", recorded_result.observation)


if __name__ == "__main__":
    unittest.main()
