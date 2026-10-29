"""The planner offers only actions supported by its target platform."""

import unittest

from yewo.agent.device.platform import Platform
from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.plan.planner import DevicePlanner


def _instructions(platform: Platform) -> str:
    planner = DevicePlanner.__new__(DevicePlanner)
    planner.grounding = GROUNDING_MARKS
    planner.platform = platform
    return planner._action_vocabulary() + planner._action_constraints()


class PlatformPlannerTests(unittest.TestCase):
    def test_ios_is_not_offered_android_key_events(self):
        instructions = _instructions(Platform.IOS)

        self.assertNotIn('"key_event"', instructions)
        self.assertNotIn("key_event 4", instructions)
        self.assertIn("platform driver", instructions)

    def test_android_is_offered_its_back_key(self):
        instructions = _instructions(Platform.ANDROID)

        self.assertIn('"key_event"', instructions)
        self.assertIn("4=back", instructions)
        self.assertIn("never to dismiss the keyboard", instructions)


if __name__ == "__main__":
    unittest.main()
